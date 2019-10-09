import argparse
import pandas as pd
import pandas.errors as pd_errors
import seaborn as sns
import os


def get_params(p: argparse.ArgumentParser) -> argparse.Namespace:
    p.add_argument('-f', '--folder', help='Folder relative to script location with CSVs to parse.', default='outputs')
    return p.parse_args()


def parse_and_gen_plots(rel_folder_path: str) -> None:
    sns.set()
    sns.set_context("notebook")

    for output in os.listdir(rel_folder_path):
        file = os.path.join(rel_folder_path, output)

        try:
            output_df = pd.read_csv(file, names=['iterations', 'train loss', 'test loss', 'accuracy'])
        except pd_errors.ParserError:
            continue

        g = sns.relplot(x='iterations', y='accuracy', kind='line', data=output_df, aspect=1.5)
        g.fig.autofmt_xdate()
        g.savefig(os.path.join(rel_folder_path, 'figures', '{fh}.png'.format(fh=output.split('.')[0])))


parser = argparse.ArgumentParser(description="Output plots for a folder of CSVs.")
args = get_params(parser)
parse_and_gen_plots(args.folder)
