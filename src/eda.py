import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def run_eda(data_path="../data/oasis_longitudinal.csv"):
    """
    Run all EDA steps.
    data_path: path to the CSV data
    """
    df = pd.read_csv(data_path)

    df = df.loc[df["Visit"] == 1]
    df = df.reset_index(drop=True)
    df["M/F"] = df["M/F"].replace(["F", "M"], [0, 1])
    df["Group"] = df["Group"].replace(["Converted"], ["Demented"])
    df["Group"] = df["Group"].replace(["Demented", "Nondemented"], [1, 0])
    df = df.drop(["MRI ID", "Visit", "Hand"], axis=1)

    def bar_chart(feature):
        Demented = df[df["Group"] == 1][feature].value_counts()
        Nondemented = df[df["Group"] == 0][feature].value_counts()
        df_bar = pd.DataFrame([Demented, Nondemented])
        df_bar.index = ["Demented", "Nondemented"]
        df_bar.plot(kind="bar", stacked=True, figsize=(8, 5))
        plt.title(f"{feature} by Demented Status")
        plt.show()

    # Gender  and  Group ( Femal=0, Male=1)
    bar_chart("M/F")
    plt.xlabel("Group")
    plt.ylabel("Number of patients")
    plt.legend()
    plt.title("Gender and Demented rate")

    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "MMSE", shade=True)
    facet.set(xlim=(0, df["MMSE"].max()))
    facet.add_legend()
    plt.xlim(15, 30)
    plt.title("MMSE Distribution by Group")
    plt.show()

    # MMSE : Mini Mental State Examination
    # Nondemented = 0, Demented =1
    # Nondemented has higher test result ranging from 25 to 30.
    # Min 17, MAX 30
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "MMSE", shade=True)
    facet.set(xlim=(0, df["MMSE"].max()))
    facet.add_legend()
    plt.xlim(15.30)

    # bar_chart('ASF') = Atlas Scaling Factor
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "ASF", shade=True)
    facet.set(xlim=(0, df["ASF"].max()))
    facet.add_legend()
    plt.xlim(0.5, 2)

    # eTIV = Estimated Total Intracranial Volume
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "eTIV", shade=True)
    facet.set(xlim=(0, df["eTIV"].max()))
    facet.add_legend()
    plt.xlim(900, 2100)

    #'nWBV' = Normalized Whole Brain Volume
    # Nondemented = 0, Demented =1
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "nWBV", shade=True)
    facet.set(xlim=(0, df["nWBV"].max()))
    facet.add_legend()
    plt.xlim(0.6, 0.9)

    # AGE. Nondemented =0, Demented =0
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "Age", shade=True)
    facet.set(xlim=(0, df["Age"].max()))
    facet.add_legend()
    plt.xlim(50, 100)

    #'EDUC' = Years of Education
    # Nondemented = 0, Demented =1
    facet = sns.FacetGrid(df, hue="Group", aspect=3)
    facet.map(sns.kdeplot, "EDUC", shade=True)
    facet.set(xlim=(df["EDUC"].min(), df["EDUC"].max()))
    facet.add_legend()
    plt.ylim(0, 0.16)

    return df

if __name__ == "__main__":
    run_eda()
