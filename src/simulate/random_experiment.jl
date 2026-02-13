function generate_random_data(fasta_wt, p, n_rounds, n_seqs; output_root)

    wt_seq = readfasta(fasta_wt)[1][2]

    variants = [wt_seq for _ in 1:n_seqs]

    for t in 1:n_rounds
        for (seq_i, seq) in pairs(variants)
            seq_aa = collect(seq)
            for i in eachindex(seq_aa)
                if rand() < p
                    old_aa_int = aa2int(seq_aa[i])
                    new_aa = int2aa(rand(collect(1:20)[1:end .!= old_aa_int]))
                    seq_aa[i] = new_aa
                end
            end

            variants[seq_i] = String(seq_aa)
        end

        open(output_root * "_rnd_$t.fasta", "w") do io
            for (i, s) in pairs(variants)
                write(io, ">seq_$(i)\n")
                write(io, s)
                write(io, "\n")
            end
        end
    end
end
