# Executive Summary — Predicting Life Expectancy from WHO Data

## The question

Life expectancy varies by more than 40 years between countries. We asked: can a
country's life expectancy be predicted from its health and economic indicators —
and which of those indicators matter most?

## The data

World Health Organization records for 193 countries, every year from 2000 to
2015 — about 2,900 country-year snapshots covering immunization rates, disease
prevalence, schooling, income, and health spending.

## What we found

**The predictions are accurate — and honestly tested.** Every model was
evaluated only on countries it had never seen during training, which is the
hard, realistic test. Our best model predicts a country's life expectancy to
within about **2 years on average**; a naive guess is off by more than 7. The
same model sorts countries into Low, Medium, or High life-expectancy groups
correctly **77% of the time** (guessing manages 34%), and when it errs it is
almost always by one group, never mistaking Low for High.

**Four indicators carry most of the signal.** HIV/AIDS prevalence and adult
mortality are the strongest predictors — unsurprising, since they measure
death directly. The interesting finding is what comes next: **education and
income composition** predict life expectancy better than most direct
health-spending measures. How educated and economically developed a population
is tells most of the rest of the story.

## What it means for health policy

- **Use the model as a screening tool.** It can flag countries whose reported
  life expectancy doesn't match what their indicators suggest — a signal to
  look closer at data quality or an unrecognized crisis.
- **Treat education and income as health policy.** In this data they are
  central to long lives, not side issues next to hospitals and spending.
- **Prioritize the known killers.** HIV/AIDS programs and adult mortality
  reduction remain the levers most tightly linked to life expectancy.

## What the model cannot do

It finds patterns, not causes — it does not prove that more schooling lengthens
lives, because richer countries have more of everything good at once. It works
at the country level, so it says nothing about any individual person. And it
reflects 2000–2015 self-reported figures, which are least reliable exactly
where health systems are weakest. Predictions for countries near a group
boundary, or unlike anything in the data, deserve extra caution.
