import os

# List of lanthanide symbols in order
lanthanides = ["LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD",
               "TB", "DY", "HO", "ER", "TM", "YB", "LU"]

# Path to your input file
input_file = "/home/users/tianyulu/code/protpardelle-1c/examples/motifs/omol/8FNS_EF2_EF3_ND_H2O.pdb"  # <-- change this to your filename

# Read entire file into a string
with open(input_file, "r") as f:
    content = f.read()

# For each lanthanide, create a modified file
for element in lanthanides:
    new_content = content.replace(" ND", f" {element}")
    new_filename = input_file.replace("_ND", f"_{element}")
    with open(new_filename, "w") as f:
        f.write(new_content)
    print(f"Saved: {new_filename}")
