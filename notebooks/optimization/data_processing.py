from sklearn.preprocessing import StandardScaler

class DataProcessor:

    @staticmethod
    def summary_categorial_data(data):
        summary_statistic = data.select_dtypes(include=['object']).describe().transpose()
        # All columns that consist of one category
        potential_columns = summary_statistic[summary_statistic["count"] == summary_statistic["unique"]].index.tolist()
        # All columns that have only unique values
        potential_columns.extend(summary_statistic[summary_statistic["unique"] == 1].index.tolist())
        return summary_statistic, potential_columns

    @staticmethod
    def scale(data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        return X_scaled

    @staticmethod
    def variance_description(data):
        variances = data.var(axis=0)
        description = variances.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50])
        return variances, description

    @staticmethod
    def remove_low_count_data(X, y):
        counts = y.value_counts()
        print(counts.sort_values())
        valid_types = counts[counts >= 2].index
        mask = y.isin(valid_types)
        return mask.values


# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
#
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(scaled_data)
# print("Number of components: ", X_pca.shape[1])
#
# explained_variance = pca.explained_variance_ratio_
#
# plt.plot(np.cumsum(explained_variance))
#
# plt.xlabel("Number of components")
# plt.ylabel("Cumulative explained variance")
# plt.show()