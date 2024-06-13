{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPydH7xDuiOy58RP6Jb83r2"
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
      "source": [
        "#install packages\n",
        "!pip install dhg"
      ],
      "metadata": {
        "id": "SE1rtAjUDC2M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "R4HeZj6BytNt"
      },
      "outputs": [],
      "source": [
        "#import packages\n",
        "import dhg\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.utils.data as Data\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "import torch.nn.functional as F\n",
        "from dhg.nn import UniGATConv\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from dhg.structure.hypergraphs import Hypergraph\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.model_selection import KFold\n",
        "from Function import train,test,data_split"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "num_cell_lines=13\n",
        "num_drugs=48"
      ],
      "metadata": {
        "id": "rQkAy2XRzAhX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "KG_cell_line_feature=pd.read_excel('*path/DATA/KG_cell_line_feature.xlsx').rename(columns={'Unnamed: 0':'cell_line_name'})\n",
        "drug_feature=pd.read_excel('*path/DATA/drug_feature.xlsx')\n",
        "extract=pd.read_excel(\"*path/DATA/extract.xlsx\")"
      ],
      "metadata": {
        "id": "iCKPUqnAzA48"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cline_feature=KG_cell_line_feature.set_index(['cell_line_name'])\n",
        "c_map = dict(zip(cline_feature.index, range(num_drugs, num_drugs + num_cell_lines)))\n",
        "\n",
        "drug_feature=drug_feature.set_index('drug_name')\n",
        "d_map = dict(zip(drug_feature.index, range(0, num_drugs)))"
      ],
      "metadata": {
        "id": "-lyBQfTc3knO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "extract_condition=extract[['drug_row','drug_col','cell_line_name','synergyscore','result']]\n",
        "syn = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3]),float(row[4])] for index, row in\n",
        "               extract_condition.iterrows() if (str(row[0]) in drug_feature.index and str(row[1]) in drug_feature.index and\n",
        "                                           str(row[2]) in cline_feature.index)]"
      ],
      "metadata": {
        "id": "hdwNRmTzCcIb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#noise_factor=0.1\n",
        "cline_feature=np.asarray(cline_feature)\n",
        "drug_feature=np.asarray(drug_feature)\n",
        "cline_feature = torch.from_numpy(cline_feature)\n",
        "#cline_feature = cline_feature + noise_factor * torch.randn_like(cline_feature)\n",
        "cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),\n",
        "                                    batch_size=len(cline_feature), shuffle=False)\n",
        "drug_feature = torch.from_numpy(drug_feature)\n",
        "#drug_feature = drug_feature + noise_factor * torch.randn_like(drug_feature)\n",
        "drug_set = Data.DataLoader(dataset=Data.TensorDataset(drug_feature),\n",
        "                                    batch_size=len(drug_feature), shuffle=False)"
      ],
      "metadata": {
        "id": "9BS981kjCyMS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4nBuoO5xr392"
      },
      "outputs": [],
      "source": [
        "class GATEncoder(torch.nn.Module):\n",
        "    def __init__(self, in_channels, out_channels):\n",
        "        super(GATEncoder, self).__init__()\n",
        "        self.conv =UniGATConv(in_channels, 256)\n",
        "        self.batch1 = nn.BatchNorm1d(256)\n",
        "        self.Linear = nn.Linear(256, out_channels)\n",
        "        self.drop_out = nn.Dropout(0.2)\n",
        "        self.act = nn.LeakyReLU(0.2)\n",
        "\n",
        "    def forward(self, x, edge):\n",
        "        x = self.act(self.conv(x, edge))\n",
        "        x = self.batch1(x)\n",
        "        x = self.drop_out(x)\n",
        "        x = self.act(self.Linear(x))\n",
        "        return x\n",
        "class BioEncoder(nn.Module):\n",
        "    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):\n",
        "        super(BioEncoder, self).__init__()\n",
        "\n",
        "        # -------cell line_layer\n",
        "        self.fc_cell1 = nn.Linear(dim_cellline, 128)\n",
        "        self.batch_cell1 = nn.BatchNorm1d(128)\n",
        "        self.fc_cell2 = nn.Linear(128, output)\n",
        "        self.drop_out = nn.Dropout(0.1)\n",
        "        self.act = nn.ReLU()\n",
        "        # -------drug line_layer\n",
        "        self.fc_drug1 = nn.Linear(dim_drug, 256)\n",
        "        self.batch_drug1 = nn.BatchNorm1d(256)\n",
        "        self.fc_drug2 = nn.Linear(256, output)\n",
        "        self.drop_out = nn.Dropout(0.2)\n",
        "        self.act = nn.ReLU()\n",
        "\n",
        "    def forward(self,gexpr_datadrug, gexpr_datacellline):\n",
        "        # -----drug_train\n",
        "        x_drug = self.fc_drug1(gexpr_datadrug.to(self.fc_drug1.weight.dtype))\n",
        "        x_drug = self.batch_drug1(x_drug)\n",
        "        x_drug = self.drop_out(x_drug)\n",
        "        x_drug = self.act(self.fc_drug2(x_drug))\n",
        "        # ----cellline_train\n",
        "        x_cellline =self.fc_cell1(gexpr_datacellline.to(self.fc_cell1.weight.dtype))\n",
        "        x_cellline = self.batch_cell1(x_cellline)\n",
        "        x_cellline = self.drop_out(x_cellline)\n",
        "        x_cellline = self.act(self.fc_cell2(x_cellline))\n",
        "        return x_drug, x_cellline\n",
        "class Decoder(torch.nn.Module):\n",
        "    def __init__(self, in_channels):\n",
        "        super(Decoder, self).__init__()\n",
        "\n",
        "        self.fc1 = nn.Linear(in_channels, 256)\n",
        "        self.batch1 = nn.BatchNorm1d(256)\n",
        "        self.fc2 = nn.Linear(256, 64)\n",
        "        self.batch2 = nn.BatchNorm1d(64)\n",
        "        self.fc3 = nn.Linear(64, 16)\n",
        "        self.batch3 = nn.BatchNorm1d(16)\n",
        "        self.fc4 = nn.Linear(16, 1)\n",
        "        self.drop_out = nn.Dropout(0.1)\n",
        "        self.act = nn.LeakyReLU(0.2)\n",
        "\n",
        "    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):\n",
        "        h = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)\n",
        "        h = h.float()  # Convert input to float\n",
        "        h = self.act(self.fc1(h))\n",
        "        h = self.batch1(h)\n",
        "        h = self.drop_out(h)\n",
        "        h = self.act(self.fc2(h))\n",
        "        h = self.batch2(h)\n",
        "        h = self.drop_out(h)\n",
        "        h=self.act(self.fc3(h))\n",
        "        h=self.batch3(h)\n",
        "        h = self.fc4(h)\n",
        "        return h.squeeze(dim=1)\n",
        "\n",
        "class HyperGraphSynergy(torch.nn.Module):\n",
        "    def __init__(self, bio_encoder, graph_encoder, decoder):\n",
        "        super(HyperGraphSynergy, self).__init__()\n",
        "        self.bio_encoder = bio_encoder\n",
        "        self.graph_encoder = graph_encoder\n",
        "        self.decoder = decoder\n",
        "\n",
        "    def forward(self,gexpr_datadrug ,gexpr_datacellline, adj, druga_id, drugb_id, cellline_id):\n",
        "        drug_embed, cellline_embed = self.bio_encoder(gexpr_datadrug , gexpr_datacellline)\n",
        "        merge_embed = torch.cat((drug_embed, cellline_embed), 0)\n",
        "        graph_embed = self.graph_encoder(merge_embed, adj)\n",
        "        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)\n",
        "        return res"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "synergy_data, test_ind, test_label=data_split(syn,rd_seed=3)\n",
        "data=synergy_data"
      ],
      "metadata": {
        "id": "dmQu1aRxET_V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "epochs =1700\n",
        "learning_rate =2e-4\n",
        "L2 = 1e-4"
      ],
      "metadata": {
        "id": "uPsG2ShUFUWl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "collapsed": true,
        "id": "Q-hsE64rE4Bl"
      },
      "outputs": [],
      "source": [
        "\n",
        "final_metric = np.zeros(3)\n",
        "fold_num = 0\n",
        "metric_values = []\n",
        "\n",
        "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n",
        "for train_index, validation_index in kf.split(data):\n",
        "    synergy_train, synergy_validation = data[train_index], data[validation_index]\n",
        "    label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32'))\n",
        "    label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32'))\n",
        "    index_train = torch.from_numpy(synergy_train[:, 0:3]).long()\n",
        "    index_validation = torch.from_numpy(synergy_validation[:, 0:3]).long()\n",
        "    # -----construct hyper_synergy_graph_set\n",
        "    synergy_train_tmp = np.copy(synergy_train)\n",
        "    my_list = synergy_train_tmp.tolist()\n",
        "    pos_edge = np.array([t for t in my_list if t[4] == 1])\n",
        "    edge_data = pos_edge[:, 0:3]\n",
        "    hyperedge=[tuple(map(int, row)) for row in edge_data]\n",
        "    hg = dhg.Hypergraph(61,hyperedge)\n",
        "    # ---model_build\n",
        "    model = HyperGraphSynergy(BioEncoder(dim_drug=drug_feature.shape[-1], dim_cellline=cline_feature.shape[-1], output=100),\n",
        "                                      GATEncoder(in_channels=100,out_channels=200), Decoder(in_channels=600))\n",
        "    loss_func = torch.nn.MSELoss()\n",
        "    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)\n",
        "\n",
        "    for epoch in range(epochs):\n",
        "        model.train()\n",
        "        train_metric, train_loss = train(drug_set, cline_set, hg,index_train, label_train)\n",
        "\n",
        "        val_metric, val_loss,_ = test(drug_set, cline_set, hg,index_validation, label_validation)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "    final_metric += val_metric\n",
        "    fold_num = fold_num + 1\n",
        "final_metric /= 5\n",
        "sd=np.std(metric_values, axis=0)\n",
        "print('Final 5-cv average results, RMSE: {:.6f},'.format(final_metric[0]),\n",
        "              'R2: {:.6f},'.format(final_metric[1]),\n",
        "              'Pearson r: {:.6f},'.format(final_metric[2]))# Concatenate the results from all folds\n",
        "\n"
      ]
    }
  ]
}