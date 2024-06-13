{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMnmIlXZ6P6Ue9i4cFBpqfh"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "naaTySHAn9CW"
      },
      "outputs": [],
      "source": [
        "#import packages\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import h5py"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "druginfo=pd.read_csv(\"*path/DATA/drugsinfo.csv\")\n",
        "KG_cell_line_feature=pd.read_excel('*path/DATA/KG_cell_line_feature.xlsx').rename(columns={'Unnamed: 0':'cell_line_name'})\n",
        "KG_drug_feature=pd.read_excel('*/path/DATA/KG_drug_feature.xlsx').rename(columns={'Unnamed: 0':'drug_name'})\n",
        "f1 = h5py.File('*path/DATA/A1_sign2.h5', 'r')# 2D_fingerprint similarty drug features\n",
        "f2 = h5py.File('*path/DATA/A5_sign2.h5', 'r')# Physicochemistry similarity drug features"
      ],
      "metadata": {
        "id": "SRS1B0oDo2s6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "spyaKgaPPTKc"
      },
      "outputs": [],
      "source": [
        "Inchkey=druginfo['InChIKey'].str.rstrip('\\xa0')\n",
        "cond=Inchkey.tolist()\n",
        "Inchkey=druginfo['InChIKey'].str.rstrip('\\xa0')\n",
        "cond=Inchkey.tolist()\n",
        "list(f1.keys())\n",
        "keys = f1['keys'][:]\n",
        "values = f1['V'][:]\n",
        "keys = [key.decode() for key in keys]\n",
        "# Create a dictionary of the data\n",
        "data_dict1 = {key: values[i] for i, key in enumerate(keys)}\n",
        "# Create a DataFrame from the dictionary\n",
        "A1= pd.DataFrame(data_dict1)\n",
        "myA1=A1.loc[:,cond]\n",
        "list(f2.keys())\n",
        "keys = f2['keys'][:]\n",
        "values = f2['V'][:]\n",
        "keys = [key.decode() for key in keys]\n",
        "# Create a dictionary of the data\n",
        "data_dict2 = {key: values[i] for i, key in enumerate(keys)}\n",
        "# Create a DataFrame from the dictionary\n",
        "A2= pd.DataFrame(data_dict2)\n",
        "myA2=A2.loc[:,cond]\n",
        "rename=dict(zip(druginfo['InChIKey'].str.rstrip('\\xa0'), druginfo['drug_name']))\n",
        "myA1=myA1.rename(columns=rename)\n",
        "myA2=myA2.rename(columns=rename)\n",
        "concat=pd.concat([myA1,myA2],axis=0)\n",
        "drug_chemchecker_feature=concat.transpose().reset_index().rename(columns={'index':'drug_name'})"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#concatenate KG_drug_feature with drug_chemchecker_feature\n",
        "drug_feature=pd.merge(drug_chemchecker_feature,KG_drug_feature,on='drug_name')\n",
        "\n",
        "#save drug_features in DATA directory"
      ],
      "metadata": {
        "id": "AZXD471SwIQ-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "9DE7CY0Rxrs2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}