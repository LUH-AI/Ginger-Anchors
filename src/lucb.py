

# https://proceedings.mlr.press/v30/Kaufmann13.pdf

# Require: eps ≥ 0 (tolerance level), U, L (confidence bounds)
# t = 1 (number of stage of the algorithm), B(1) = ∞ (stopping index)
# for a=1...K do
    # Sample arm a, compute confidence bounds Ua(1), La(1)
# end for
# while B(t) > eps do
    # Draw arm ut and lt. t = t + 1.
    # Update confidence bounds, set J(t) and arms ut, lt
    # B(t) = Uut (t) − Llt (t)
# end while
# return J(t).

u = max()