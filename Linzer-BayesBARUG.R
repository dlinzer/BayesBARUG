## 
## Drew Linzer
## drew@votamatic.org
## @DrewLinzer
## November 12, 2013
## 
## Linzer-BayesBARUG.R
## 
## Doing Bayesian statistics in R
## Bay Area useR Group November 2013 meetup
## 

## Why be Bayesian?
## (Or at least, why add Bayesian methods to your toolbox?)
## It's philosophically satisfying: the logic comports with intuition
## It's creatively liberating: model whatever you want, however you want
## It's focused on the world: makes analysis about understanding, not significance testing
## -- Jacob Cohen (1994) "The Earth Is Round (p < .05)" http://bit.ly/1bDGRwk
## -- Alex Reinhart "Statistics Done Wrong" http://www.refsmmat.com/statistics/index.html

## R has tons of excellent packages for Bayesian data analysis.
##   CRAN Task View: Bayesian Inference
##   http://cran.r-project.org/web/views/Bayesian.html

library(R.utils)
library(rstan)
library(R2jags)
library(manipulate)
library(MCMCpack)
library(R2WinBUGS)

## Bayes in a nutshell: Three equivalent statements
## What we think about the world after seeing data =
##   What we thought about the world before seeing data x
##   Chance we'd see our data under different assumptions about the world
## Pr(world|data) = Pr(world) x Pr(data|world)
## Posterior = Prior x Likelihood
## A posterior is a probability distribution, used for inference.

## A simple example: Estimating a proportion from dichotomous 0/1 data

## For the prior distribution, choose "hyperparameters" that describe our 
##   belief about a quantity of interest (here, p) before seeing data.
## The Beta distribution is "conjugate" to the binomial likelihood, 
##   with hyperparameters alpha and beta.

p <- seq(from=0.005, to=0.995, by=0.005)

manipulate( # requires RStudio
  {plot(p, dbeta(p, alpha.hyper, beta.hyper), 
       col="blue", lwd=2, type="l", las=1, bty="n", 
       ylim=c(0, 8), ylab="density", 
       main="Beta prior distribution")
  polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper, beta.hyper), 
       rep(0, length(p))), col=rgb(0, 0, 1, 0.2), border=NA)}, 
  alpha.hyper=slider(0.1, 10, step=0.1, initial=1), 
  beta.hyper=slider(0.1, 10, step=0.1, initial=1))

## Now we observe some data
p.true <- 0.7
N <- 30
y <- rbinom(N, size=1, prob=p.true)
table(y)/N

## Likelihood of the data at each possible value of p
## (http://en.wikipedia.org/wiki/Bernoulli_distribution)
likelihood <- sapply(p, function(p) { prod(p^y * (1-p)^(1-y)) } )
plot(p, likelihood, lwd=2, las=1, bty="n", type="l")

## (To help with visibility)
like.rescale <- 4 * likelihood/max(likelihood)

## To get the posterior, multiply Prior x Likelihood at each value of p
## Or easier: Prior is conjugate, so posterior is Beta distributed with
##   alpha = alpha + k
##   beta = beta + N - k
## Where N = sample size, k = number of "successes".
## The prior is most influential when data are sparse.

manipulate(
  {plot(p, like.rescale, lwd=2, las=1, bty="n", 
        ylim=c(0,8), type="l", ylab="density", 
        main="Beta prior (blue) x Likelihood (black) = Beta posterior (red)")
   alpha.hyper.post <- alpha.hyper + sum(y)
   beta.hyper.post <- beta.hyper + N - sum(y)
   lines(p, dbeta(p, alpha.hyper, beta.hyper), col="blue", lwd=2)
   polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper, beta.hyper), 
                           rep(0, length(p))), col=rgb(0, 0, 1, 0.2), border=NA)
   lines(p, dbeta(p, alpha.hyper.post, beta.hyper.post), col="red", lwd=2)
   polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper.post, beta.hyper.post), 
                           rep(0, length(p))), col=rgb(1, 0, 0, 0.2), border=NA)
   lines(p, like.rescale, lwd=2)}, 
   alpha.hyper=slider(0.1, 10, step=0.1, initial=1), 
   beta.hyper=slider(0.1, 10, step=0.1, initial=1))

## Inferences about the quantity of interest come from the posterior:
##   A best guess (posterior mean or mode), as well as a statement of
##   uncertainty, typically expressed as a "credible interval" or range
##   representing percent area of highest posterior density.


####################################################################################
####################################################################################

## What if an analytical solution isn't practical?
## Use fast computers and Markov chain Monte Carlo (MCMC) sampling
##   to find the shape of the posterior distribution.

## JAGS = "Just Another Gibbs Sampler" to take random draws from the posterior
## Very similar to BUGS = "Bayesian inference Using Gibbs Sampling"

# Specify the Beta-Bernoulli model
jags.bb <- function() {
  for (i in 1:N) {
    y[i] ~ dbin(p, 1)
  }
  # prior on p
  p ~ dbeta(alpha.hyper, beta.hyper)
}

write.model(jags.bb, "jagsbb.txt")
# file.show("jagsbb.txt")

# Specify hyperpriors and initial values
alpha.hyper <- 1
beta.hyper <- 1
inits <- function() { list(p=runif(1)) }

# Fit the model
jagsfit <- jags.parallel(data=c("y", "N", "alpha.hyper", "beta.hyper"), 
                         inits=inits, 
                         parameters.to.save=c("p"), 
                         model.file="jagsbb.txt", 
                         n.chains=3,
                         n.iter=10000)
jagsfit
plot(jagsfit)
traceplot(jagsfit, mfrow=c(1,1), "p")

## There are additional MCMC diagnostics in the coda package
jagsfit.mcmc <- as.mcmc(jagsfit)
xyplot(jagsfit.mcmc)
densityplot(jagsfit.mcmc)
autocorr.plot(jagsfit.mcmc)
codamenu()

## Posterior distribution of parameters of interest
jags.mod <- jagsfit$BUGSoutput
names(jags.mod)
jags.mod$mean
hist(jags.mod$sims.list$p, breaks=30, col="gray", xlim=c(0,1), main="", freq=F)

# Should match analytical solution:
alpha.hyper.post <- alpha.hyper + sum(y)
beta.hyper.post <- beta.hyper + N - sum(y)
lines(p, dbeta(p, alpha.hyper.post, beta.hyper.post), col="red", lwd=2)


####################################################################################
####################################################################################

## Stan is a much newer (mid-2012) software package for MCMC estimation,
##   still under development. It's an alternative for big or complex
##   models that may be slow to converge in JAGS or BUGS.

?stan

# The Beta-Bernoulli model
stan.bb <- "data {
    int<lower=0> N;
    int<lower=0,upper=1> y[N];
  }
  parameters {
    real<lower=0,upper=1> p;
  }
  model {
    p ~ beta(1,1);
    y ~ bernoulli(p);
  }"

stanfit <- stan(model_code=stan.bb,
                 data=list(y=y, N=N), 
                 iter=10000,
                 chains=3)
  
hist(extract(stanfit)$p, breaks=30, col="gray", xlim=c(0,1), main="", freq=F)

# Should again match analytical solution:
lines(p, dbeta(p, alpha.hyper.post, beta.hyper.post), col="red", lwd=2)


####################################################################################
####################################################################################

##
## Example II: Forecasting as a missing data problem
##

## Who's going to win the 2016 presidential election?
## Data for Abramowitz "Time-for-Change" forecasting model:
##   incumbent.vote: Vote share of incumbent party candidate, major parties only
##   gdp.growth: Percent increase in GDP, Q1 to Q2 of election year
##   net.approval: President's June approval-disapproval rating from Gallup
##   two.terms: Has the incumbent party held the presidency for 2 or more terms?
## Note the NAs for the 2016 election

dat <- data.frame(year = c(2016,2012,2008,2004,2000,1996,1992,1988,1984,1980,1976,1972,1968,1964,1960,1956,1952,1948),
                  gdp.growth = c(NA,1.3,1.3,2.6,8,7.1,4.3,5.2,7.1,-7.9,3,9.8,7,4.7,-1.9,3.2,0.4,7.5),
                  net.approval = c(NA,-0.8,-37,-0.5,19.5,15.5,-18,10,20,-21.7,5,26,-5,60.3,37,53.5,-27,-6),
                  two.terms = c(1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,1,1),
                  incumbent.vote = c(NA,52.0,46.3,51.2,50.3,54.7,46.5,53.9,59.2,44.7,48.9,61.8,49.6,61.3,49.9,57.8,44.5,52.4))
nrow(dat)
splom(dat[,-1])

## Could simply run a linear regression... To forecast, have to guess
##   Obama approval, GDP growth in mid-2016; hard to figure uncertainty.
reg <- lm(incumbent.vote ~ gdp.growth + net.approval + two.terms, dat)
summary(reg)
coefficients(reg) %*% c(1,2,0,1) # (for example)

## Instead, write out a full probability model for JAGS

jags.lm <- function() {
  for (i in 1:N) {
    incumbent.vote[i] ~ dnorm(mu[i], tau)
    mu[i] <- b[1] + b[2]*gdp.growth[i] + b[3]*net.approval[i] + b[4]*two.terms[i]
  }
  # priors for missing 2016 x's: historical mean, precision (1/variance)
  gdp.growth[1] ~ dnorm(3.7, 0.05)
  net.approval[1] ~ dnorm(7.7, 0.002)
  # non-informative priors on b's
  for (j in 1:4) {
    b[j] ~ dnorm(0, 0.001)
  }  
  # conditional variance of y given x
  tau <- pow(sd, -2)
  sd ~ dunif(0, 100)
}

write.model(jags.lm, "jagslm.txt")
# file.show("jagslm.txt")

## Fit the model
attachLocally(dat)
N <- nrow(dat)
inits <- function() { list(b=c(50, rnorm(3))) }
jagsfit <- jags.parallel(data=c("incumbent.vote", "gdp.growth",
                                "net.approval", "two.terms", "N"), 
                         inits=inits, 
                         parameters.to.save=c("b", "sd", "incumbent.vote", 
                                "gdp.growth", "net.approval"), 
                         model.file="jagslm.txt", 
                         n.chains=3, 
                         n.iter=10000)

## Look at the model output, check convergence.
plot(jagsfit)
traceplot(jagsfit, mfrow=c(1,1), "incumbent.vote")

jags.mod <- jagsfit$BUGSoutput
jags.mod$mean$b                  # regression coefficients
jags.mod$mean$incumbent.vote[1]  # forecast of 2016

## Probability the Democrat will win more than 50% of the vote?
## That's the proportion of simulated draws greater than 50.
dim(jags.mod$sims.list$incumbent.vote)
hist(jags.mod$sims.list$incumbent.vote[,1], breaks=50, col="gray",
     xlab="Democrat's predicted two-party vote share", main="")
table(jags.mod$sims.list$incumbent.vote[,1]>50) / jags.mod$n.sims

## Can change this prediction by tinkering with priors on the b's or 2016 x's...

  
####################################################################################
####################################################################################

## What if you just wanted to run a regression, but be Bayesian?
## Many common models can be found in the package MCMCpack.
help(package="MCMCpack")

?MCMCregress
breg <- MCMCregress(incumbent.vote ~ gdp.growth + net.approval + two.terms, dat)
summary(breg)
plot(breg)


####################################################################################
####################################################################################

## For more information:

## JAGS
## ====
## http://mcmc-jags.sourceforge.net/
## http://jeromyanglim.blogspot.com/2012/04/getting-started-with-jags-rjags-and.html

## BUGS
## ====
## www.mrc-bsu.cam.ac.uk/bugs/

## Stan
## ====
## http://mc-stan.org/
## https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started

## MCMCpack
## ========
## http://mcmcpack.wustl.edu/
## http://www.jstatsoft.org/v42/i09/

## Books
## =====
## David Lunn et al. "The BUGS Book: A Practical Introduction to 
##   Bayesian Analysis" (2012)
##   http://www.mrc-bsu.cam.ac.uk/bugs/thebugsbook/
## John K. Kruschke "Doing Bayesian Data Analysis: A Tutorial 
##   with R and BUGS" (2010)
##   http://www.indiana.edu/~kruschke/DoingBayesianDataAnalysis/



## end of file