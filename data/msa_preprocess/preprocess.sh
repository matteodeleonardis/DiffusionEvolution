#!/bin/bash

set -x

PROT_NAME="$1"
PROT_FILE="$2"
NAT_FILE="$3"
OUT_DIR="$4"

# Full header without '>'
read -r prot_name < "$PROT_FILE"
prot_name=${prot_name#>}

# Sequence length (handles multiline FASTA, stops at next entry)
prot_length=$(awk '
  /^>/ {if (seq) exit; next}
  {seq = seq $0}
  END {print length(seq)}
' "$PROT_FILE")

jackhmmer \
    -A ${PROT_NAME}.sto \
    -o ${PROT_NAME}.out.txt \
	"$PROT_FILE" "$NAT_FILE"

./sto2fasta_order ${PROT_NAME}.sto > ${PROT_NAME}.fasta 

python process.py "${PROT_NAME}" "${prot_length}" "${PROT_NAME}.fasta" "${OUT_DIR}"