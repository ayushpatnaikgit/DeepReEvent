library(splines)
library(survival)


#=====================================================================
#
#                         'CindexRec' function
#
#=====================================================================
#
# CindexRec(formula, data0, CovEst=FALSE, n.boot=100, seedn=78496)
#
#---------------------------------------------------------------------
# Description:
#------------- 
# Calculate concordance index (C-index) with recurrent events data 
# to evaluation the discriminatory power of proportional means/rates models (Lin et al., 2000),
# or a frailty model.  
#
# Arguments:
#-----------
# formula -- a formula object. The response must be a survival object as returned by the Surv function.
# data -- a data.frame in which to interpret the variables named in the formula
# CovEst -- logical value; if TRUE, the variance of C-index is estimated using a resampling-based method.
# n.boot -- the number of perturbation re-samples for variance estimation
# seedn -- seed number for re-sampling
#
# Required library: survival
#=====================================================================



###################################################################
#
#	Functions
#
###################################################################
EstCstat <- function(score, N, Cis, RE, nRE){ 
  den = 0;
  num = 0;
  for (i in 1: (N-1)){	 
    ID1 = Cis[i, 1]
    ## Cis[i, 2]) == common observation time		
    search.event <- which(RE[ ,3] <= Cis[i, 2])
    if (length(search.event) > 0) {
      if (length(search.event)==1) {
    	ID.COUNT = RE[search.event,1]
    	COUNT <- matrix(1,1,1)
      } else{ 	
    	REtemp = RE[search.event, ]     	
    	COUNT <- as.matrix(table(REtemp[ ,1]))   
     	ID.COUNT <- as.numeric(rownames(COUNT))
      }
      nREc = rep(0, N)
      nREc[ID.COUNT] <- COUNT[ ,1]
      IDpair <- sort(Cis[(i+1):N, 1])   ### same as: IDpair=which(Ci >= Cis[i, 2])
      nREc <- nREc[IDpair]

      lt_obs = ifelse(nRE[ID1] < nREc, 1, 0)
      gt_obs = ifelse(nRE[ID1] > nREc, 1, 0)
      lt_pred <- ifelse(score[ID1] < score[IDpair], 1, 0) 
      gt_pred <- ifelse(score[ID1] > score[IDpair], 1, 0)  

      den = den + sum(lt_obs + gt_obs)	
      num = num + sum(lt_obs*lt_pred + gt_obs*gt_pred)
    }  ## end of if(length(search.event) > 0)
  } ## end of for
  estcstat = num/den;   c(estcstat, den);
}

CindextRec <- function(formula, data, CovEst=FALSE, n.boot=100, seedn=78496){
  data <- data.frame(data)
  Call <- match.call()
  indx <- match(c("formula", "data"), names(Call), nomatch=0) 
	    if (indx[1] ==0) stop("A formula argument is required")
  temp <- Call[c(1,indx)]  # only keep the arguments we wanted
  temp[[1L]] <- quote(stats::model.frame)  # change the function called
  special <- c("cluster", "frailty")
  temp$formula <- if(missing(data)) terms(formula, special)
                    else              terms(formula, special, data=data)
  mf <- eval(temp, parent.frame())
    if (nrow(mf) ==0) stop("No (non-missing) observations")
  Terms <- terms(mf)
    cluster<- attr(Terms, "specials")$cluster
    frailty<- attr(Terms, "specials")$frailty

  fit <- coxph(formula, data=data)
  coxph0 <- summary(fit)
  newbeta = fit$coefficients
  nbetas = length(newbeta)
  
  #######################################################
  #	Data Steps: create Zi, Ci, RE
  #######################################################  
  if(length(cluster)){search.term='cluster'} else if(length(frailty)){
	search.term='frailty'
  } else{
	stop("'cluster' or 'frailty' should be part of 'formula' argument")
  }	
  X = model.matrix(formula, data)[ ,-1]
  pos.id <- which(regexpr(search.term, colnames(X))==1)
  data.id <- X[, pos.id]       ## or, you can use model.frame
  X = as.matrix(X[ ,-pos.id]);    rownames(X)<- data.id;
  n.record <- GetLastID(data.id)
  last.id <- cumsum(n.record) 
  Zi = as.matrix(X[last.id, ]);   
  N = length(last.id)    # number of indep. persons

  mf = model.frame(fit, data=data)
  Y = model.extract(mf, "response");   rownames(Y)<- data.id;
  Ci = Y[last.id, 2]
  PID <- 1:N
  ci <- cbind(PID, Ci)
  Cis = ci[order(ci[ ,2]), ]	  ### Cis = sorted Ci

  status1 = which(Y[ ,3]==1)
  status1.id <- data.id[status1]
  IDbook <- cbind(PID, data.id[last.id])
  PID.RE = rep(0, length(status1.id))
  nRE = rep(0, N)
  for (person in 1:N) {
    search.id <- which(status1.id==IDbook[person,2])  
    PID.RE[search.id] = IDbook[person,1];	
    nRE[person] = length(search.id)   
  }   
  RE = cbind(PID.RE, status1.id, Y[status1, 2])

  #######################################################
  #	C-stat estimation
  ####################################################### 
#  pred = c(Zi %*% newbeta)
  pred = fit$linear.predictors[last.id]
  Ecstat <- EstCstat(pred, N, Cis, RE, nRE)
    estcstat = Ecstat[1]
    npair = Ecstat[2]
    estout = c(newbeta,estcstat)
    names(estout) <- c(names(newbeta),'C-stat')

if (length(frailty)){
  CovEst=FALSE
  specials='frailty'
  out=list(est=estout, 
			n.subjet=N, n.pair=npair, n.rec.event=nRE, Call=Call, CovEst=CovEst, 
			output.coxph=coxph0, specials=specials)
}
if (length(cluster)){
  specials='marginal'
  se.robust <- sqrt(diag(fit$var))

  out=list(est=estout, se.robust.coxph=se.robust,
			n.subjet=N, n.pair=npair, n.rec.event=nRE, Call=Call, CovEst=CovEst,
			output.coxph=coxph0, specials=specials)
    
  if (CovEst==TRUE){  
  #######################################################
  #	Variance calculation using perturbation (Resampling)
  #######################################################
  VV = matrix(0, N, N)
  II = matrix(0, N, N)
  for (i in 1:(N-1)){	 
     ID1 = Cis[i, 1]
     search.event <- which(RE[ ,3] <= Cis[i, 2])
     if (length(search.event) > 0) {
      if (length(search.event)==1) {
    	ID.COUNTr = RE[search.event,1]
    	COUNTr <- matrix(1,1,1)
      } else{ 	
    	REr = RE[search.event, ]     	
    	COUNTr <- as.matrix(table(REr[ ,1]))   
     	ID.COUNTr <- as.numeric(rownames(COUNTr))
      }
      nREr = rep(0, N)
      nREr[ID.COUNTr] <- COUNTr[ ,1]
      IDcomp <- sort(Cis[(i+1):N, 1])   ### same as: IDpair=which(Ci >= Cis[i, 2])
      nREr <- nREr[IDcomp]
      r.lt_obs = ifelse(nRE[ID1] < nREr, 1, 0)
      r.gt_obs = ifelse(nRE[ID1] > nREr, 1, 0)
      r.lt_pred <- ifelse(pred[ID1] < pred[IDcomp], 1, 0) 
      r.gt_pred <- ifelse(pred[ID1] > pred[IDcomp], 1, 0)   
      V_ij <- r.lt_obs*(r.lt_pred-estcstat) + r.gt_obs*(r.gt_pred-estcstat)  
      I_ij <- r.lt_obs  + r.gt_obs 
          
      ind.upper<- which(ID1 < IDcomp)
      ind.lower <- which(ID1 > IDcomp)
      VV[ID1, IDcomp[ind.upper]] <- V_ij[ind.upper]
      VV[IDcomp[ind.lower],ID1] <- V_ij[ind.lower]
      II[ID1, IDcomp[ind.upper]] <- I_ij[ind.upper]
      II[IDcomp[ind.lower],ID1] <- I_ij[ind.lower]
      }  ## end of if(length(search.event) > 0)
  }     ## end of for

  inv.Ahat = N*fit$naive.var
  assign('data', data, envir = .GlobalEnv)
  resid.score = resid(fit, type='score', collapse=data.id)
   
  sim_r=1;
  estpar_r <- NULL
  WW_all <- NULL
  set.seed(seedn)
  while (sim_r <= n.boot){
      rs=rexp(N,1)
	 	
      ### V-perturbation & pi-perturbation
      s.V =0 
      r.npair =0
      for (i in 1: (N-1)){	 	
	     s.V = s.V + sum(VV[i, (i+1):N]*rs[i]*rs[(i+1):N])
	     r.npair = r.npair + sum(II[i, (i+1):N]*rs[i]*rs[(i+1):N])
      }     	 
      ### beta-perturbation
      s.resid.score = 0
      for (i in 1:(N-1)){
	     s.resid.score = s.resid.score + colSums((matrix(resid.score[i, ],N-i,nbetas,byrow=TRUE) + resid.score[(i+1):N, ])*rs[i]*rs[(i+1):N])/2	
      } 
      r.beta= newbeta + 2/(N*(N-1))*inv.Ahat%*%s.resid.score
      ### c-stat with beta-perturbated 
      r.pred = c(Zi %*% r.beta)
      r.Ecstat <- EstCstat(r.pred, N, Cis, RE, nRE)
         
      ### NOTE: 2/(N*(N-1)) in the 1st term cancelled out
      WW1 =  s.V/npair
      WW2 = (r.Ecstat[1]-estcstat)      
      WW =  WW1 + WW2   
      estpar_r = rbind(estpar_r, c(r.Ecstat[1], r.beta))   
      WW_all = rbind(WW_all, c(WW, WW1, WW2))
      sim_r = sim_r + 1;
  }

  bias_r = t(estpar_r) - colMeans(estpar_r)
  sd_r = sqrt(diag(bias_r%*%t(bias_r)) /(nrow(estpar_r)-1))
  bias_WW = t(WW_all) - colMeans(WW_all)
  sd_WW = sqrt(diag(bias_WW%*%t(bias_WW)) /(nrow(WW_all)-1)) 

  out=list(est=estout, se.robust.coxph=se.robust,
			se.resample.coxph=sd_r[-1], se.cstat=sd_WW[1], 
			n.subjet=N, n.pair=npair, n.rec.event=nRE, Call=Call, CovEst=CovEst, 
			output.coxph=coxph0, specials=specials)
  } 

} 
class(out) <- 'CindexRec'
out
}


###################################################################
#
#	Examples
#
###################################################################
## Fit a marginal model with with CovEst==FALSE (default)
fit1 <- CindexRec(Surv(Tstart,Tstop,Status) ~  Trt + Number + cluster(id), data=one)

## Fit a marginal model with CovEst==TRUE 
fit2 <- CindexRec(Surv(Tstart,Tstop,Status) ~  Trt + Number + cluster(id), data=one, CovEst=TRUE)

## Fit a frailty model with CovEst==FALSE. Currently, CovEst=TRUE is not available.
fit3 <- CstatRec(Surv(Tstart,Tstop,Status) ~  Trt + Number + frailty(id), data=one)