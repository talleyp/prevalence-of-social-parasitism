# # All rates per day
# ## Transition rates
# rq = 24.   # queen egg laying rate
# re = 1/14  # egg hatching rate
# rl = 1/7   # pupation rate
# rp = 1/23  # rate of pupae to nurse
# rn = 0#1/365 # rate of becoming a forager

# ## death rates
# muf = 1/60 # death rate of forager

# ## Regulation terms
# Km = 1
# Kf = 1/25.

data = {
        'c': 5,
        'rq': 12,
        're': 1/14.,
        'rl': 1/7.,
        'rn': 0,
        'rp': 1/23.,
        'rs': 0,#0.43,
        'A': .308,
        'mun': 1/7,
        'muf': 1/60.,
        'Km': 1.,
        'Kf': 1/100.,
        'prd': 300,
        'cst': True
    }