class DataProcessor:

    @staticmethod
    def summary_categorial_data(data):
        summary_statistic = data.select_dtypes(include=['object']).describe().transpose()
        # All columns that consist of one category
        potential_columns = summary_statistic[summary_statistic["count"] == summary_statistic["unique"]].index.tolist()
        # All columns that have only unique values
        potential_columns.extend(summary_statistic[summary_statistic["unique"] == 1].index.tolist())
        return summary_statistic, potential_columns