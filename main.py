#!/usr/bin/python3.6
from typing import List, Tuple, DefaultDict, Callable, TypeVar
from itertools import islice
from collections import defaultdict
from collections import Counter
from heapq import nlargest
from operator import itemgetter
import csv
import math
import random
import pickle
import numpy as np

T = TypeVar('T')
U = TypeVar('U')


UserType = str
ItemType = str
DataSet = List[Tuple[UserType, ItemType, float]]
RecommendFunction = Callable[[UserType, ItemType], float]
UserToItem = DefaultDict[UserType, DefaultDict[ItemType, float]]


def load_data(filename: str = "data/train_triplets.txt", lines: int = -1) -> DataSet:
    with open(filename) as file:
        data = csv.reader(file, delimiter='\t')
        if lines > 0:
            data = islice(data, lines)
        data = [(user, song, float(count)) for (user, song, count) in data]
    return data


def split_data(data: DataSet, train_coef: float = 0.8) -> Tuple[DataSet, DataSet]:
    assert 0. <= train_coef <= 1.

    train_len = int(train_coef * len(data))
    train_indexes = set(random.sample(range(len(data)), k=train_len))
    train, test = [], []
    for i in range(len(data)):
        if i in train_indexes:
            train.append(data[i])
        else:
            test.append(data[i])
    return train, test


class Recommender:
    def init_with_user_to_item(self, u2i: DefaultDict[UserType, DefaultDict[ItemType, float]]):
        self.user_to_item = u2i
        self.item_to_user = Recommender._transpose(self.user_to_item)
        self.user_to_mean = self._user_to_mean()
        self.item_to_mean = self._item_to_mean()
        self.user_to_deviation = self._user_to_deviation()
        self.item_to_deviation = self._item_to_deviation()

    def __init__(self, train: DataSet):
        self.user_to_item = None
        self.item_to_user = None
        self.user_to_mean = None
        self.item_to_mean = None
        self.user_to_deviation = None
        self.item_to_deviation = None
        self.init_with_user_to_item(Recommender._user_to_item(train))

    @staticmethod
    def _user_to_item(ds: DataSet) -> UserToItem:
        u2i2r = defaultdict(lambda: defaultdict(float))
        for user, item, count in ds:
            u2i2r[user][item] += count
        return u2i2r

    @staticmethod
    def _transpose(t2u: DefaultDict[T, DefaultDict[U, float]]) -> DefaultDict[U, DefaultDict[T, float]]:
        u2t = defaultdict(lambda: defaultdict(float))
        for x in t2u:
            t2u_x = t2u[x]
            for y in t2u_x:
                u2t[y][x] += t2u_x[y]
        return u2t

    def _user_to_mean(self) -> DefaultDict[UserType, float]:
        u2m = defaultdict(float)
        for user in self.user_to_item:
            u2m[user] = sum(self.user_to_item[user].values()) / len(self.user_to_item[user])
        return u2m

    def _item_to_mean(self) -> DefaultDict[ItemType, float]:
        i2m = defaultdict(float)
        for item in self.item_to_user:
            i2m[item] = sum(self.item_to_user[item].values()) / len(self.item_to_user[item])
        return i2m

    def _user_to_deviation(self) -> DefaultDict[UserType, float]:
        return defaultdict(float)

    def _item_to_deviation(self) -> DefaultDict[ItemType, float]:
        return defaultdict(float)

    def CV(self, user_u: UserType, user_v: UserType) -> float:
        def total_items(user: UserType) -> float:
            s = 0.
            for count in self.user_to_item[user].values():
                s += count ** 2
            return s ** .5

        s = 0.
        items_u = self.user_to_item[user_u]
        items_v = self.user_to_item[user_v]
        for item in set(items_u) & set(items_v):
            s += items_u[item] * items_v[item]
        if s == 0.:
            return s
        return s / (total_items(user_u) * total_items(user_v))
        # return math.fabs(s / (total_items(user_u) * total_items(user_v)))

    def PC_user(self, user_u: UserType, user_v: UserType) -> float:
        def total_items(user: UserType) -> float:
            s = 0.
            mean = self.user_to_mean[user]
            for count in self.user_to_item[user].values():
                s += (count - mean) ** 2
            return s ** .5

        s = 0.
        items_u = self.user_to_item[user_u]
        items_v = self.user_to_item[user_v]
        mean_u = self.user_to_mean[user_u]
        mean_v = self.user_to_mean[user_v]
        for item in set(items_u) & set(items_v):
            s += (items_u[item] - mean_u) * (items_v[item] - mean_v)
        if s == 0.:
            return s
        return s / (total_items(user_u) * total_items(user_v))
        # return math.fabs(s / (total_items(user_u) * total_items(user_v)))

    def _user_vector_similarity_recommend(self, similarity: Callable[[UserType, UserType], float], user: UserType, item: ItemType, k: int = 1) -> List[Tuple[UserType, float]]:
        users = self.item_to_user[item]
        but_this_users = users.keys() - {user}
        if not but_this_users:
            return []
        users_with_similarities = [(user, similarity(user, other_user)) for other_user in but_this_users]
        best_users = nlargest(k, users_with_similarities, key=lambda t: t[1])
        return best_users

    def neighborhood_recommend_mean(self, similarity: Callable[[UserType, UserType], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_users = self._user_vector_similarity_recommend(similarity, user, item, k=k)
        ratings = [r for _, r in best_users]
        return sum(ratings) / len(ratings) if ratings else 0.

    def neighborhood_recommend_weight(self, similarity: Callable[[UserType, UserType], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_users = self._user_vector_similarity_recommend(similarity, user, item, k=k)
        if not best_users:
            return 0.
        weight_rating = 0.
        total_rating = 0.
        users = self.item_to_user[item]
        for other_user, sim in best_users:
            weight_rating += sim * users[other_user]
            total_rating += math.fabs(sim)
        if total_rating == 0.:
            return total_rating
        return weight_rating / total_rating

    def PC_item(self, item_i: ItemType, item_j: ItemType) -> float:
        def total_items(item: UserType) -> float:
            s = 0.
            mean = self.item_to_mean[item]
            for count in self.item_to_user[item].values():
                s += (count - mean) ** 2
            return s ** .5

        s = 0.
        users_i = self.item_to_user[item_i]
        users_j = self.item_to_user[item_j]
        mean_u = self.item_to_mean[item_i]
        mean_v = self.item_to_mean[item_j]
        for user in set(users_i) & set(users_j):
            s += (users_i[user] - mean_u) * (users_j[user] - mean_v)
        if s == 0.:
            return s
        return math.fabs(s / (total_items(item_i) * total_items(item_j)))

    def AC(self, item_i: ItemType, item_j: ItemType) -> float:
        user_intersection = set(self.item_to_user[item_i]) & set(self.item_to_user[item_j])

        def total_items(item: ItemType) -> float:
            s = 0.
            users = self.item_to_user[item]
            for user in user_intersection:
                s += (users[user] - self.user_to_mean[user]) ** 2
            return s ** .5

        s = 0.
        users_i = self.item_to_user[item_i]
        users_j = self.item_to_user[item_j]
        for user in user_intersection:
            mean = self.user_to_mean[user]
            s += (users_i[user] - mean) * (users_j[user] - mean)
        if s == 0.:
            return s
        return math.fabs(s / (total_items(item_i) * total_items(item_j)))

    def _item_vector_similarity_recommend(self, similarity: Callable[[ItemType, ItemType], float], user: UserType, item: ItemType, k: int = 1) -> List[Tuple[ItemType, float]]:
        items = self.user_to_item[user]
        but_this_items = items.keys() - {item}
        if not but_this_items:
            return []
        items_with_similarities = [(item, similarity(item, other_item)) for other_item in but_this_items]
        best_items = nlargest(k, items_with_similarities, key=lambda t: t[1])
        return best_items

    def item_recommend_mean(self, similarity: Callable[[ItemType, ItemType], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_items = self._item_vector_similarity_recommend(similarity, user, item, k=k)
        ratings = [r for _, r in best_items]
        return sum(ratings) / len(ratings) if ratings else 0.

    def item_recommend_weight(self, similarity: Callable[[ItemType, ItemType], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_items = self._item_vector_similarity_recommend(similarity, user, item, k=k)
        if not best_items:
            return 0.
        weight_rating = 0.
        total_rating = 0.
        items = self.user_to_item[user]
        for other_item, sim in best_items:
            weight_rating += sim * items[other_item]
            total_rating += math.fabs(sim)
        if total_rating == 0.:
            return total_rating
        return weight_rating / total_rating


class MeanCenteredRecommender(Recommender):
    def _mean_center(self, similarity: Callable[[T, T], float], user: UserType, item: ItemType, k: int = 1) -> float:
        raise NotImplementedError

    def __init__(self, train: DataSet, similarity: str = "CV", k: int = 1):
        super(MeanCenteredRecommender, self).__init__(train)

        if similarity == "CV":
            similarity = self.CV
        elif similarity == "PC_user":
            similarity = self.PC_user
        elif similarity == "PC_item":
            similarity = self.PC_item
        elif similarity == "AC":
            similarity = self.AC
        else:
            raise AttributeError("No such similarity option: %s" % similarity)

        user_to_item = defaultdict(lambda: defaultdict(float))
        old_user_to_item = self.user_to_item.copy()
        for user in old_user_to_item:
            for item in self.user_to_item[user]:
                user_to_item[user][item] = self._mean_center(similarity, user, item, k=k)

        self.init_with_user_to_item(user_to_item)


class UserBasedMeanCenteredRecommender(MeanCenteredRecommender):
    def _mean_center(self, similarity: Callable[[T, T], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_users = self._user_vector_similarity_recommend(similarity, user, item, k=k)
        weight_rating = 0.
        total_rating = 0.
        users = self.item_to_user[item]
        for other_user, sim in best_users:
            weight_rating += sim * (users[other_user] - self.user_to_mean[other_user])
            total_rating += math.fabs(sim)
        rating = self.user_to_mean[user]
        return rating + (0. if total_rating == 0. else weight_rating / total_rating)


class ItemBasedMeanCenteredRecommender(MeanCenteredRecommender):
    def _mean_center(self, similarity: Callable[[T, T], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_items = self._item_vector_similarity_recommend(similarity, user, item, k=k)
        weight_rating = 0.
        total_rating = 0.
        items = self.user_to_item[user]
        for other_item, sim in best_items:
            weight_rating += sim * (items[other_item] - self.item_to_mean[other_item])
            total_rating += math.fabs(sim)
        rating = self.item_to_mean[item]
        return rating + (0. if total_rating == 0. else weight_rating / total_rating)


class ZscoreNormalizedRecommender(MeanCenteredRecommender):
    def _user_to_deviation(self) -> DefaultDict[UserType, float]:
        u2d = defaultdict(float)
        for user in self.user_to_item:
            s = 0.
            mean = self.user_to_mean[user]
            for rating in self.user_to_item[user].values():
                s += (rating - mean) ** 2
            u2d[user] = 1. if s == 0. else (s / (len(self.user_to_item[user]) - 1)) ** .5

        return u2d

    def _item_to_deviation(self) -> DefaultDict[ItemType, float]:
        i2d = defaultdict(float)
        for item in self.item_to_user:
            s = 0.
            mean = self.item_to_mean[item]
            for rating in self.item_to_user[item].values():
                s += (rating - mean) ** 2
            i2d[item] = 1. if s == 0. else (s / (len(self.item_to_user[item]) - 1)) ** .5

        return i2d


class UserBasedZscoreNormalizedRecommender(ZscoreNormalizedRecommender):
    def _mean_center(self, similarity: Callable[[UserType, UserType], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_users = self._user_vector_similarity_recommend(similarity, user, item, k=k)
        weight_rating = 0.
        total_rating = 0.
        users = self.item_to_user[item]
        for other_user, sim in best_users:
            weight_rating += sim * (users[other_user] - self.user_to_mean[other_user]) / self.user_to_deviation[other_user]
            total_rating += math.fabs(sim)
        rating = self.user_to_mean[user]
        return rating + (0. if total_rating == 0. else self.user_to_deviation[user] * weight_rating / total_rating)


class ItemBasedZscoreNormalizedRecommender(ZscoreNormalizedRecommender):
    def _mean_center(self, similarity: Callable[[T, T], float], user: UserType, item: ItemType, k: int = 1) -> float:
        best_items = self._item_vector_similarity_recommend(similarity, user, item, k=k)
        weight_rating = 0.
        total_rating = 0.
        items = self.user_to_item[user]
        for other_item, sim in best_items:
            weight_rating += sim * (items[other_item] - self.item_to_mean[other_item]) / self.item_to_deviation[other_item]
            total_rating += math.fabs(sim)
        rating = self.item_to_mean[item]
        return rating + (0. if total_rating == 0. else self.item_to_deviation[item] * weight_rating / total_rating)


def RMSE(test: DataSet, recommend: RecommendFunction):
    s = 0.
    for u, i, r in test:
        s += (recommend(u, i) - r) ** 2
    return (s / len(test)) ** .5


def MAE(test: DataSet, recommend: RecommendFunction):
    s = 0.
    for u, i, r in test:
        s += abs(recommend(u, i) - r)
    return s / len(test)


def dcg_at_p(r, p):
    r = np.asfarray(r)[:p]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_p(r, p):
    dcg_max = dcg_at_p(sorted(r, reverse=True), p)
    if not dcg_max:
        return 0.
    return dcg_at_p(r, p) / dcg_max


def split_data_between_users(test: DataSet):
    dict = defaultdict(list)
    for (u, i, r) in test:
        dict[u].append((i, r))
    return dict


def get_item(iar, i):
    for item, r in iar:
        if item == i:
            return r
    return 0

def nDCGp(test: DataSet, recommend: RecommendFunction, p):  # Mean nDCG between all users
    s = 0.
    dict = split_data_between_users(test)
    items = set([i for _, i, _ in test])
    for user, iar in dict.items():
        ratings = [(recommend(user, item), get_item(iar, item)) for item in items]
        sorted_ratings = list(map(itemgetter(1), sorted(ratings, key=itemgetter(0), reverse=True)))
        s += ndcg_at_p(sorted_ratings, p)
    return s / len(dict)


def gini_coefficient(test: DataSet, recommend: RecommendFunction, p):
    users = set([u for u, _, _ in test])
    items = set([i for _, i, _ in test])
    items_frequency = Counter()
    for i in items:
        items_frequency[i] = 0
    for user in users:
        ratings = [(recommend(user, item), item) for item in items]
        topN = map(itemgetter(1), sorted(ratings, key=itemgetter(0), reverse=True)[:p])
        for i in topN:
            items_frequency[i] += 1
    s = 0.
    n = len(items)
    values = items_frequency.values()
    for j, frequency in enumerate(sorted(values, key=lambda x: x, reverse=False)):
        s += (2 * j - n - 1) * frequency
    return s / ((n - 1) * sum(values))


def main():
    def load_saved():
        with open("data/ds20000.pickle", 'rb') as file:
            return pickle.load(file)

    K = 5  # Top-K
    if True:
        data = load_saved()
    else:
        data = load_data(lines=2000)
        with open("data/ds20000.pickle", 'wb') as file:
            pickle.dump(data, file)

    train, test = split_data(data)
    r = Recommender(train)  # Base Class
    # r = UserBasedMeanCenteredRecommender(train, similarity="CV", k=K)
    # r = UserBasedMeanCenteredRecommender(train, similarity="AC", k=K)
    # r = ItemBasedMeanCenteredRecommender(train, similarity="CV", k=K)
    # r = ItemBasedMeanCenteredRecommender(train, similarity="AC", k=K)
    # r = ItemBasedZscoreNormalizedRecommender(train, similarity="CV", k=K)

    with open("results.txt", 'w+') as file:
        file.write("RMSE for all implemented recommender functions\n")
        file.write(str(RMSE(test, lambda u, i: r.neighborhood_recommend_mean(r.CV, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.neighborhood_recommend_mean(r.PC_user, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.neighborhood_recommend_weight(r.CV, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.neighborhood_recommend_weight(r.PC_user, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.item_recommend_mean(r.PC_item, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.item_recommend_mean(r.AC, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.item_recommend_weight(r.PC_item, u, i, k=K))) + "\n")
        file.write(str(RMSE(test, lambda u, i: r.item_recommend_weight(r.AC, u, i, k=K))) + "\n")

        file.write("MAE for all implemented recommender functions\n")
        file.write(str(MAE(test, lambda u, i: r.neighborhood_recommend_mean(r.CV, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.neighborhood_recommend_mean(r.PC_user, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.neighborhood_recommend_weight(r.CV, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.neighborhood_recommend_weight(r.PC_user, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.item_recommend_mean(r.PC_item, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.item_recommend_mean(r.AC, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.item_recommend_weight(r.PC_item, u, i, k=K))) + "\n")
        file.write(str(MAE(test, lambda u, i: r.item_recommend_weight(r.AC, u, i, k=K))) + "\n")

        file.write("nDCG for all implemented recommender functions\n")
        file.write(str(nDCGp(test, lambda u, i: r.neighborhood_recommend_mean(r.CV, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.neighborhood_recommend_mean(r.PC_user, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.neighborhood_recommend_weight(r.CV, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.neighborhood_recommend_weight(r.PC_user, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.item_recommend_mean(r.PC_item, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.item_recommend_mean(r.AC, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.item_recommend_weight(r.PC_item, u, i, k=K), K)) + "\n")
        file.write(str(nDCGp(test, lambda u, i: r.item_recommend_weight(r.AC, u, i, k=K), K)) + "\n")

        file.write("Gini coefficient for all implemented recommender functions\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.neighborhood_recommend_mean(r.CV, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.neighborhood_recommend_mean(r.PC_user, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.neighborhood_recommend_weight(r.CV, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.neighborhood_recommend_weight(r.PC_user, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.item_recommend_mean(r.PC_item, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.item_recommend_mean(r.AC, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.item_recommend_weight(r.PC_item, u, i, k=K), K)) + "\n")
        file.write(str(gini_coefficient(test, lambda u, i: r.item_recommend_weight(r.AC, u, i, k=K), K)) + "\n")


if __name__ == '__main__':
    main()
