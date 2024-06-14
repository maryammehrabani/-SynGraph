{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNkmDZmRYgl2hgqgt9Y+2bq"
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
        "from scipy.stats import pearsonr"
      ],
      "metadata": {
        "id": "S2U3WOLcLpgz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x5I536GfMOdk"
      },
      "outputs": [],
      "source": [
        "def tanh_normalize(x):\n",
        "    return np.tanh(x)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Vxt5RgQuokLj"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score\n",
        "def regression_metric(ytrue, ypred):\n",
        "    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)\n",
        "    r2 = r2_score(y_true=ytrue, y_pred=ypred)\n",
        "    r, p = pearsonr(ytrue, ypred)\n",
        "    return rmse, r2, r"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "tQuoCQmyNK5U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EeDkEaUGyut5"
      },
      "outputs": [],
      "source": [
        "def cosine_similarity(arr1, arr2):\n",
        "    dot_product = np.dot(arr1, arr2)\n",
        "    norm1 = np.linalg.norm(arr1)\n",
        "    norm2 = np.linalg.norm(arr2)\n",
        "    similarity = dot_product / (norm1 * norm2)\n",
        "    return similarity"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "bBKHoHkuNboD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yTNue1aDNKXE"
      },
      "outputs": [],
      "source": [
        "def train(drug_feature_set, cline_feature_set, synergy_adj, index, label):\n",
        "    loss_train = 0\n",
        "    true_ls, pre_ls = [], []\n",
        "    optimizer.zero_grad()\n",
        "    for batch, (drug, cline) in enumerate(zip(drug_feature_set, cline_feature_set)):\n",
        "        pred= model(drug[0], cline[0], synergy_adj,index[:, 0], index[:, 1], index[:, 2])\n",
        "        loss = loss_func(pred, label)\n",
        "        loss =  loss\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        loss_train += loss.item()\n",
        "        true_ls += label_train.cpu().detach().numpy().tolist()\n",
        "        pre_ls += pred.cpu().detach().numpy().tolist()\n",
        "    rmse, r2, pr = regression_metric(true_ls, pre_ls)\n",
        "    return [rmse, r2, pr], loss_train\n",
        "def test(drug_feature_set, cline_feature_set, synergy_adj, index, label):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        for batch, (drug, cline) in enumerate(zip(drug_feature_set, cline_feature_set)):\n",
        "            pred = model(drug[0], cline[0], synergy_adj, index[:, 0], index[:, 1], index[:, 2])\n",
        "        loss = loss_func(pred, label)\n",
        "        loss = loss\n",
        "        rmse, r2, pr = regression_metric(label.cpu().detach().numpy(),\n",
        "                                         pred.cpu().detach().numpy())\n",
        "        return [rmse, r2, pr], loss.item(),pred.cpu().detach().numpy()"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Dr8FeZnNDYva"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1CUCTy5eOfDD"
      },
      "outputs": [],
      "source": [
        "def data_split(synergy, rd_seed=3):\n",
        "    synergy_pos = pd.DataFrame([i for i in syn])\n",
        "\n",
        "    # -----split synergy into 5CV,test set\n",
        "    train_size = 0.9\n",
        "    synergy_data, synergy_test = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),\n",
        "                                             [int(train_size * len(synergy_pos))])\n",
        "\n",
        "    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32'))\n",
        "    test_ind = torch.from_numpy(np.array(synergy_test[:, 0:3])).long()\n",
        "    return synergy_data, test_ind, test_label"
      ]
    }
  ]
}