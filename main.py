import pandas as pd
import src.simple_func as simple_func
import datetime
import src.big_func as big_func
import argparse

# Création de l'objet parser
parser = argparse.ArgumentParser(
    description="Exécute des opérations sur des données en utilisant des paramètres spécifiques.")

# Ajout des arguments
parser.add_argument('--var1', default="classe1", help='Description pour variable1')
parser.add_argument('--var2', default="HLA-A", help='Description pour variable2')
parser.add_argument('--input', default="data/exemple.csv", help='Fichier d\'entrée CSV')
parser.add_argument('--run-name', default=datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
                    help='Nom de la session en cours')

args = parser.parse_args()

# Utilisation des arguments argsparse pour construire les chemins des répertoires et fichiers
output_directory = f"results/{args.run_name}/{args.var1}_{args.var2}.csv"
figure_directory = f"figures/{args.run_name}/{args.var1}_{args.var2}.png"

if __name__ == "__main__":
    # Make run_name subdir in results and figures
    simple_func.make_results_folder(args.run_name)

    # Load data
    df = pd.read_csv(args.input, sep=";")

    # apply big func on df
    df = big_func.big_func(df, args.var1, args.var2)

    # Save df to csv
    df.to_csv(output_directory, sep=";", index=False)
