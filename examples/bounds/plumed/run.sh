#!/bin/bash
cd cos_3D

COORDS="0.3 0.3 0.3"

run_simulation() {
  local pfile=$1

cat > input_md.dat << EOF
nstep         1000000000
tstep         0.001
temperature   1.0
friction      10.0
dimension     3
ipos          $COORDS
plumed        $pfile
periodic true
min -pi -pi -pi
max  pi  pi  pi
idum    637245470
EOF

   plumed pesmd input_md.dat > log
}


echo "Starting iteration 0..."
run_simulation "plumed.dat"
mv COLVAR COLVAR_0


for i in $(seq 1 12 ); do
  echo "Starting iteration $i..."

  COORDS=$(tail -n 1 COLVAR_$((i-1)) | awk '{print $2" "$3" "$4}')

  echo "RESTART" > plumed_cont.dat
  cat plumed.dat >> plumed_cont.dat

  run_simulation "plumed_cont.dat"
  mv COLVAR COLVAR_$i
done

rm -f plumed_cont.dat

HILLS_COUNT=$(grep -c -v '^[[:space:]]*#' HILLS)
echo "Total hills recorded: $HILLS_COUNT"

sed -i 's/"total_steps": [0-9]*/"total_steps": '$HILLS_COUNT'/' job_config.json
metad_readhills
metad_target_distr
cd ../

