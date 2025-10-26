from datasets import load_dataset
import pandas as pd

def info_dataset(nome, dataframes):
    df = pd.concat(dataframes)
    print(f"{nome.upper()}:")
    for split, d in dataframes.items():
        print(f"{split:>12}: {len(d)} samples")
    print(f"Total: {len(df)} samples")
    print(f"Columns: {list(df.columns)}\n")

def analisar_todos():
    ds = load_dataset('glue', 'sst2')
    info_dataset('sst2', {'train': pd.DataFrame(ds['train']),
                          'validation': pd.DataFrame(ds['validation'])})

    ds = load_dataset('glue', 'cola')
    info_dataset('cola', {'train': pd.DataFrame(ds['train']),
                          'validation': pd.DataFrame(ds['validation'])})

    ds = load_dataset('glue', 'mrpc')
    info_dataset('mrpc', {'train': pd.DataFrame(ds['train']),
                          'validation': pd.DataFrame(ds['validation']),
                          'test': pd.DataFrame(ds['test'])})

    ds = load_dataset('glue', 'rte')
    info_dataset('rte', {'train': pd.DataFrame(ds['train']),
                         'validation': pd.DataFrame(ds['validation'])})

    ds = load_dataset('super_glue', 'boolq')
    info_dataset('boolq', {'train': pd.DataFrame(ds['train']),
                           'validation': pd.DataFrame(ds['validation'])})

    ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
    info_dataset('trec', {'train': pd.DataFrame(ds['train']),
                          'test': pd.DataFrame(ds['test'])})

    ds = load_dataset('ag_news')
    info_dataset('ag_news', {'train': pd.DataFrame(ds['train']),
                             'test': pd.DataFrame(ds['test'])})

    ds = load_dataset('benayas/snips')
    info_dataset('snips', {'train': pd.DataFrame(ds['train']),
                           'test': pd.DataFrame(ds['test'])})

    ds = load_dataset('fancyzhx/dbpedia_14')
    info_dataset('db_pedia', {'train': pd.DataFrame(ds['train']),
                              'test': pd.DataFrame(ds['test'])})

if __name__ == "__main__":
    analisar_todos()
