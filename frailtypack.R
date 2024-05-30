library(frailtypack)

# Load the data
data(readmission)

# Set seed for reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets
train_indices <- sample(seq_len(nrow(readmission)), size = 0.8 * nrow(readmission))
train_data <- readmission[train_indices, ]
test_data <- readmission[-train_indices, ]

# Build the joint frailty model on the training set using the gap time approach
modJoint.gap <- frailtyPenal(
  Surv(time, event) ~ cluster(id) + sex + dukes + charlson + terminal(death),
  formula.terminalEvent = ~ sex + dukes + charlson,
  data = train_data,
  n.knots = 14,
  kappa = c(9.55e+9, 1.41e+12),
  recurrentAG = FALSE
)

# Summary of the model
summary(modJoint.gap)

