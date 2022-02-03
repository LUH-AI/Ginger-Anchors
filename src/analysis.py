
# for different 𝐵, 𝛿 and 𝜖 collect:
#   - runtime + n_samples
#   - precision
#   - coverage   
#   - bounds
# -> handle timeouts
#
# nice to have -> samples, precision and bounds per arm for one run

B = [1, 2, 3, 4, 5]
delta = [0.05, 0.1, 0.15, 0.2, 0.25]
# delta bigger -> beta smaller -> bounds less far from mean -> more confident in our sampled precision
epsilon = [0.1, 0.15, 0.2, 0.25, 0.3]

# analysing BO

# number of evaluations -> coverage, precision
# quantization factor

