data(sleepstudy, package = 'lme4')
library(data.table)
library(ggplot2)
library(rstan)
library(bayesplot)
library(lme4)
library(flexmix)


s2 <- as.data.table(sleepstudy)

s2[, subject_ind := 1:10, by = 'Subject']

s2[, Reaction := Reaction + 30 * subject_ind] # make these subjects worse

s2[, Reaction := Reaction + rnorm(.N, 0, 30)] # add noise to make these new subjects different
s2[, Subject := as.factor(as.numeric(levels(Subject))[Subject] + 1000)]
s2[, label := 'drug']
s2$subject_ind <- NULL

s1 <- as.data.table(sleepstudy)
s1[, label := 'nodrug']

new_data <- rbind(s1, s2)

data <- list(N = nrow(new_data),
             y = new_data$Reaction,
             x = new_data$Days,
             nsub = length(levels(new_data$Subject)),
             nobs_sub = c(new_data[,.N, by = Subject][,2])[[1]])

mod <- stan_model('test_latentclass.stan')

fit <- sampling(mod, data = data,
                chains = 3, cores = 3,
                iter = 3000)

print(fit, pars = c('intercept', 'slope', 'u', 'sigma_u', 'sigma_e'))

print(fit, pars = c('lambda', 'sim_group')) # estimated group belongingship

y_sim <- extract(fit, 'y_sim')[[1]]
ppc_ribbon_grouped(new_data$Reaction, y_sim, x = new_data$Days, group = new_data$Subject)

group_sim <- extract(fit, 'sim_group')[[1]]
stan_clusters <- rep(colMeans(group_sim), each = 10)

# reference models (divide by known labels)
lmer_1 <- lmer(Reaction ~ Days + (1|Subject), data = new_data[label == 'drug'])
lmer_2 <- update(lmer_1, data = new_data[label == 'nodrug'])

ggplot(new_data, aes(x = Days, y = Reaction, colour = label, group = Subject)) + geom_line()

## TODO: flexmix example (should be able to handle it?)
flex_mod <- flexmix(. ~ .|Subject, k = 2, 
               model = FLXMRlmer(Reaction ~ Days, random = ~1), 
               data = new_data)

xyplot(Reaction ~ Days | clusters(flex_mod), groups = Subject, data = new_data, type = 'l')
xyplot(colMeans(y_sim) ~ Days | round(stan_clusters), groups = Subject, data = new_data, type = 'l')
