# Hackathon Problem 1: Machine Damage Accumulation Prediction using Heterogeneous Temporal Sensor Data

## Problem Statement
The Bernard M. Gordon Learning Factory is a hands-on facility for engineering students, which provides modern design, prototyping, and manufacturing facilities. Many of the machines in the Learning Factory are instrumented using a sensor suite that provides monitoring capabilities. The readings from a heterogeneous set of sensors is used to report metrics continuously for many different machines.

The objective of this problem is to forecast future usage patterns for five commonly used machines in the learning factory with a time resolution of 10 minutes. Machine usage, here, is a binary variable where a true value indicates that a machine will be used at some point during a future 10 minute interval, and a false value indicates that the machine will not be used during that interval. Specifically, teams should train algorithms that are capable of predicting usage for up to 2 hours in the future: 12 binary values per machine, each representing whether or not the machine will be used in successive 10-minute intervals.

## Challenges
This data and use case present several challenges. These include:

- Is it feasible to construct a data-driven digital twin for forecasting? (Kunath et al., 2018)
- What is the best approach to identifying appropriate signals in this data with which to make predictions? (Long et al., 2019)
- The data provided here represents only the technical side of the system, but is heavily impacted by the human use of the learning factory as well. (Smith et al., 2013).
- What is an appropriate threshold on measured sensor values to indicate active use of the machine? (Yan et al., 2017)
