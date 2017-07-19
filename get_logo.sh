for i in {0..2}
do
	echo $i
	weblogo -f 'W_conv1_layer'$i'.txt' -D transfac -A dna -o 'w'$i'.png' -F png --color red A 'A' --color green T 'T' --color blue C 'C' --color yellow G 'G' -s large --resolution 300
done
