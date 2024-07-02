function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end