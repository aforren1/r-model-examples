data {
  int<lower=0> N; // total number of observations
  real y[N]; // response
  int<lower=0> nsub; // number of subjects
  int<lower=1> nobs_sub[nsub]; // number of observations per subject
  real x[N]; // continuous predictor
}

transformed data {
}
parameters {
  vector[2] intercept; // enforce ordering to help identification
  ordered[2] slope;
  
  real u[nsub]; // random intercept per subject
  real<lower=0> sigma_u; // variance for random effect
  real<lower=0> sigma_e; // residual variance
  real<lower=0, upper=1> lambda[nsub]; // soft ownership prob per subject
}

transformed parameters {
}
model {
  int n = 1;
  vector[2] tmp_lpdf[N];
  intercept ~ normal(250, 10);
  slope ~ normal(20, 5);
  sigma_u ~ normal(30, 5);
  sigma_e ~ normal(30, 5);
  u ~ normal(0, sigma_u);
  lambda ~ beta(1, 1);

  for (a in 1:nsub) {
      for (b in 1:nobs_sub[a]) {
        tmp_lpdf[n, 1] = normal_lpdf(y[n] | intercept[1] + u[a] + slope[1] * x[n], 
                                     sigma_e);
        tmp_lpdf[n, 2] = normal_lpdf(y[n] | intercept[2] + u[a] + slope[2] * x[n],
                                     sigma_e);
        target += log_mix(lambda[a], tmp_lpdf[n, 1], tmp_lpdf[n, 2]);
        n = n + 1;
      }
  }
}

generated quantities {
  //real y_sim[N]; // simulated vals
  // figure out group, then simulate from proper group
}
