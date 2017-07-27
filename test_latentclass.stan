data {
  int<lower=0> N; // total number of observations
  real y[N]; // response (reaction time)
  int<lower=0> nsub; // number of subjects
  int<lower=1> nobs_sub[nsub]; // number of observations per subject
  real x[N]; // continuous predictor (days of sleep deprivation?)
}

parameters {
  vector[2] intercept; 
  ordered[2] slope; // enforce ordering to help identification; try as vector to see if there's problem
  vector[2] u[nsub]; // random intercept per subject
  vector<lower=0>[2] sigma_u; // variance for random effect
  vector<lower=0>[2] sigma_e; // residual variance
  real<lower=0, upper=1> lambda[nsub]; // ownership prob per subject
}

model {
  int n = 1; // counter
  vector[2] lpdf_parts[N];
  
  // priors
  // population-level effects
  intercept ~ normal(250, 10);
  slope ~ normal(20, 5);
  
  sigma_u ~ normal(30, 5); // group-level variance
  sigma_e ~ normal(30, 5); // residual variance
  u[:, 1] ~ normal(0, sigma_u[1]); // group-level estimates (like conditional modes in frequentist)
  u[:, 2] ~ normal(0, sigma_u[2]);
  lambda ~ beta(1, 1); // flat prior over probability of membership

  for (a in 1:nsub) {
      for (b in 1:nobs_sub[a]) {
        lpdf_parts[n, 1] = normal_lpdf(y[n] | intercept[1] + u[a, 1] + slope[1] * x[n], sigma_e[1]);
        lpdf_parts[n, 2] = normal_lpdf(y[n] | intercept[2] + u[a, 2] + slope[2] * x[n], sigma_e[2]);
        // log_mix does log_sum_exp(log(lambda) + normal_lpdf(), log1m(lambda) + normal_lpdf())
        target += log_mix(lambda[a], lpdf_parts[n, 1], lpdf_parts[n, 2]);
        n = n + 1;
      }
  }
}

generated quantities {
  real y_sim[N]; // simulated vals
  real sim_group[nsub]; // simulated group membership (0 or 1)
  int n = 1; // counter, redux

  // figure out group membership
  for (i in 1:nsub) {
      vector[2] lpdf_per_sub;
      vector[2] prob;
      
      lpdf_per_sub[1] = 0;
      lpdf_per_sub[2] = 0;

      for (j in 1:nobs_sub[i]) {
          lpdf_per_sub[1] = lpdf_per_sub[1] + normal_lpdf(y[n] | intercept[1] + u[i, 1] + slope[1] * x[n], sigma_e[1]);
          lpdf_per_sub[2] = lpdf_per_sub[2] + normal_lpdf(y[n] | intercept[2] + u[i, 2] + slope[2] * x[n], sigma_e[2]);
          n = n + 1;
      }
      lpdf_per_sub[1] = lpdf_per_sub[1] + log(lambda[i]);
      lpdf_per_sub[2] = lpdf_per_sub[2] + log1m(lambda[i]);
      prob = softmax(lpdf_per_sub);
      sim_group[i] = bernoulli_rng(prob[1]);
  }
  n = 1;
  
  // generate simulated values
  for (i in 1:nsub) {
      for (j in 1:nobs_sub[i]) {
        if (sim_group[i] > 0) {
          y_sim[n] = normal_rng(intercept[1] + u[i, 1] + slope[1] * x[n], sigma_e[1]);
        } else {
          y_sim[n] = normal_rng(intercept[2] + u[i, 2] + slope[2] * x[n], sigma_e[2]);    
        }
        n = n + 1;
      }
  }
}
