#!/bin/bash

for file in *.ipynb
do
    # Get the file prefix (i.e., strip the .ipynb extension)
    filename="${file%.ipynb}"
    # Convert the notebook to a script
    jupyter nbconvert --to script --output "${filename}.py" "$file"
    mv "${filename}.py.py" "${filename}.py"
    mv "${filename}.py.txt" "${filename}.py"
done

# run with conda run -n ginaenv python AliaMolecule_Adversarially_Guided_Diffusion.py
