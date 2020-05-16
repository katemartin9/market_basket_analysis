import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# MovieLens dataset (100k) can be downloaded from https://grouplens.org/datasets/movielens/


class MBA:
    def __init__(self, title):
        self.output_df = pd.DataFrame()
        self.title = title
        self.b_users = None
        self.length = None

    def probability_of_item(self, x):
        """This method calculates a probability (frequency of each item"""
        quantity = len(x)
        return quantity / self.length

    def lift_calculation(self, x):
        """This method calculates lift - P(A&B) / P(A)/P(B)"""
        return x[0]/x[1]

    def a_and_b_prob(self, x):
        """This method calculates P(A&B) - probability of two items being together
        in one basket (aka support)"""
        intercected_values = np.intersect1d(x, self.b_users, assume_unique=True)
        return len(intercected_values) / self.length

    def a_mult_b_prob(self, x):
        """This method calculates P(A)*P(B)"""
        prob_b = self.output_df[self.output_df.title == self.title]['P(A)'].item()
        return x*prob_b

    def forming_baskets(self, df):
        """Forming baskets and calculating support, confidence and lift"""
        self.length = df['user_id'].nunique()
        self.output_df = df.groupby('title')['user_id'].apply(list).to_frame().reset_index()
        # calculating frequancy of each item in the dataframe - P(A)
        self.output_df['P(A)'] = self.output_df['user_id'].apply(lambda x: self.probability_of_item(x))
        self.output_df = self.output_df[self.output_df['P(A)'] > 0.02]  # filtering out all infrequent items
        self.output_df['title_B'] = self.title
        self.b_users = self.output_df[self.output_df.title == self.title]['user_id'].item()
        # calculating probability of title A and title B being in the same basket
        self.output_df['P(A&B)'] = self.output_df['user_id'].apply(lambda x: self.a_and_b_prob(x))
        # multiplying probabilities of title A and title B
        self.output_df['P(A)P(B)'] = self.output_df['P(A)'].apply(lambda x: self.a_mult_b_prob(x))
        # calculating lift
        self.output_df['lift'] = self.output_df[['P(A&B)', 'P(A)P(B)']].apply(self.lift_calculation, axis=1)


if __name__ == '__main__':
    df_user = pd.read_csv('u.data', sep='\t', header=None).iloc[:, :2]
    df_movie = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None).iloc[:, :2]
    df_user.columns = ['user_id', 'movie_id']
    df_movie.columns = ['movie_id', 'title']
    df_mba = df_user.merge(df_movie, on='movie_id', how='left')
    df_mba = df_mba.drop_duplicates()

    mba = MBA('Snow White and the Seven Dwarfs (1937)')  # passing in a movie title to find assosiations
    mba.forming_baskets(df_mba)
    my_df = mba.output_df.sort_values(by='lift', ascending=False)
    print(my_df.head(10))




