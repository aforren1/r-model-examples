data(sleepstudy, package = 'lme4')
library(data.table)
library(ggplot2)
library(rstan)
library(lme4)


s2 <- as.data.table(sleepstudy)

s2[, subject_ind := 1:10, by = 'Subject']

s2[, Reaction := Reaction + 40 * subject_ind] # make these subjects worse

s2[, Reaction := Reaction + rnorm(.N, 0, 30)]
s2[, Subject := as.factor(as.numeric(levels(Subject))[Subject] + 1000)]
s2[, label := 'drug']
s2$subject_ind <- NULL

s1 <- as.data.table(sleepstudy)
s1[, label := 'nodrug']

new_data <- rbind(s1, s2)

unique_ids <- as.numeric(levels(new_data$Subject))
for (ii in 1:length(unique_ids)) {
    new_data$subject2[which(new_data$Subject == unique_ids[ii])] <- ii
}


data <- list(N = nrow(new_data),
             y = new_data$Reaction,
             x = new_data$Days,
             nsub = length(levels(new_data$Subject)),
             nobs_sub = c(new_data[,.N, by = Subject][,2])[[1]])

mod <- stan_model('test_latentclass.stan')

fit <- sampling(mod, data = data,
                chains = 3, cores = 3,
                iter = 3000)

print(fit, pars = c('intercept', 'slope', 'u', 'sigma_u', 'sigma_e', 'lambda'))

# half of our stan model should roughly match this
m_lme4 <- lmer(Reaction ~ Days + (1|Subject), data = sleepstudy)

ggplot(new_data, aes(x = Days, y = Reaction, colour = label, group = Subject)) + geom_line()
