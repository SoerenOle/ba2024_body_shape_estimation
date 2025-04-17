import sys
import csv
from itertools import product
from pathlib import Path

def main():    
    combinationsTable = create_standard_table()
    relative_file_path = "Data/table_combinations/"
    file_path = Path(__file__).parents[1] / relative_file_path
    write_csv(combinationsTable, file_path, "combinations_table.csv")

def create_standard_table():
    #value1 = Gender
    #value2 = Age
    #value3 = Muscle
    #value4 = Weight
    #value5 = Height
    #value6 = Proportions

    #Alle Altersgruppen ausser "Babys"
    p1 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p2 = [0.25, 0.5, 0.75, 1.0]
    p3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p5 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p6 = [0.0, 0.25, 0.5, 0.75, 1.0]

    #Babys ... Modelle existieren nur wenn P6 = 0.5 
    q1 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q2 = [0.0]
    q3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q5 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q6 = [0.5]

    #Kombinationen von p1-p6 und q1-q6 bilden
    combinations_p = list(product(p1, p2, p3, p4, p5, p6, repeat=1))
    combinations_q = list(product(q1, q2, q3, q4, q5, q6, repeat=1))

    #Tabellen zusammenführen
    combinations = combinations_q + combinations_p
    return combinations

def create_breast_param_table():
    #value1 = Gender
    #value2 = Age
    #value3 = Muscle
    #value4 = Weight
    #value5 = Height
    #value6 = Proportions
    #value7 = Cup Size
    #value8 = Brest Firmness

    #Nur Frauen & alle Altersgruppen ausser "Babys"
    p1 = [0.0, 0.25]
    p2 = [0.25, 0.5, 0.75, 1.0]
    p3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p5 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p6 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p7 = [0.0, 0.25, 0.5, 0.75, 1.0]
    p8 = [0.0, 0.25, 0.5, 0.75, 1.0]

    #Neutral + Männer & alle Altersgruppen ausser "Babys"
    q1 = [0.5, 0.75, 1.0]
    q2 = [0.25, 0.5, 0.75, 1.0]
    q3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q5 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q6 = [0.0, 0.25, 0.5, 0.75, 1.0]
    q7 = [9]
    q8 = [9]

    #Nur Frauen & Babys ... Modelle existieren nur wenn P6 = 0.5 
    r1 = [0.0, 0.25, 0.5, 0.75, 1.0]
    r2 = [0.0]
    r3 = [0.0, 0.25, 0.5, 0.75, 1.0]
    r4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    r5 = [0.0, 0.25, 0.5, 0.75, 1.0]
    r6 = [0.5]
    r7 = [9]
    r8 = [9]

    #Kombinationen von p1-p8, q1-q6 und r1-r6 bilden
    combinations_p = list(product(p1, p2, p3, p4, p5, p6, p7, p8, repeat=1))
    combinations_q = list(product(q1, q2, q3, q4, q5, q6, repeat=1))
    combinations_r = list(product(r1, r2, r3, r4, r5, r6, repeat=1))

    #Tabellen zusammenführen
    combinations = combinations_q + combinations_p + combinations_r
    return combinations

def write_csv(combinations, path, name):
    # define output file
    file_path = Path(path)
    file_name = Path(name)
    output_file = file_path / file_name
    print(len(combinations))
    #file_path = "./Data/table_combinations/"
    #name = combinations_table_1.csv"
    file_path.mkdir(parents=True, exist_ok=True)
    with output_file.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        # writer.writerow(["id", "p1", "p2", "p3", "p4", "p5", "p6"])
        # Write the rows with ID
        for idx, combo in enumerate(combinations):
            writer.writerow([idx] + list(combo))

    print(f"Table generated and saved to {output_file}") 

if __name__ == '__main__':
    sys.exit(main()) 

