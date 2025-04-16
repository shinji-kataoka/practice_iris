# クラスに落とし込む
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import ward
import pandas as pd
import matplotlib.pyplot as plt
from ClassVisualizer import Visualizer
import numpy as np


class AnalyzeIris:
    def __init__(self):
        """初期化メソッド"""
        self.iris_data = load_iris()
        self.df = pd.DataFrame(
            data=self.iris_data.data,
            columns=self.iris_data.feature_names
        )
        self.df['label'] = self.iris_data.target
        self.x = self.df.drop('label', axis=1)
        self.y = self.df['label']
        self.feature_names = self.x.columns
        self.models = [
            LogisticRegression(max_iter=1000),
            LinearSVC(max_iter=5000, dual=False),
            DecisionTreeClassifier(),
            KNeighborsClassifier(n_neighbors=4),
            LinearRegression(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            MLPClassifier(max_iter=1000)]
        self.plot_models = [
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier()
        ]
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        self.scaler = [
            MinMaxScaler(),
            StandardScaler(),
            RobustScaler(),
            Normalizer()
            ]
        self.label_names = self.iris_data.target_names
        self.visualizer = Visualizer()

    def Get(self):
        """データフレームを取得するメソッド
        Returns:
            pd.DataFrame: データフレーム
        """
        return self.df

    def PairPlot(
        self,
        cmap='brg',
        figsize=(15, 15),
        marker='o',
        hist_kwds=None,
        pointsize=60,
        alpha=.8
    ):
        """散布図行列をプロットするメソッド
        Args:
            cmap (str, optional): カラーマップ. Defaults to 'brg'.
            figsize (tuple, optional): 図のサイズ. Defaults to (15, 15).
            marker (str, optional): マーカーの形. Defaults to 'o'.
            hist_kwds (dict, optional): ヒストグラムのキーワード引数. Defaults to None.
            pointsize (int, optional): ポイントサイズ. Defaults to 60.
            alpha (float, optional): 透明度. Defaults to .8.
        Returns:
            pd.plotting.scatter_matrix: 散布図行列
        """
        return pd.plotting.scatter_matrix(
            self.x,
            c=self.y,
            figsize=figsize,
            marker=marker,
            hist_kwds=hist_kwds,
            s=pointsize,
            alpha=alpha,
            cmap=cmap)

    def AllSupervised(self, n_neighbors=4):
        """全ての手法の交差検証を行い、スコアを出力するメソッド
        Args:
            n_neighbors (int, optional): KNeighborsClassifierの近傍数. Defaults to 4.
        Returns:
            None
        """
        X, y = self.x, self.y
        kfold = self.kfold
        models = self.models
        self.models_index = [x.__class__.__name__ for x in models]
        # 全ての手法のスコアを保存するリスト
        self.train_scores_list = []
        self.test_scores_list = []
        for model in models:
            # 各手法の交差検証のスコアを保存するリスト
            train_scores = []
            test_scores = []
            for fold, (train_index, test_index) in enumerate(kfold.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # モデルの訓練
                model.fit(X_train, y_train)
                # 訓練スコアとテストスコアの計算
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                # 結果をリストに追加
                train_scores.append(train_score)
                test_scores.append(test_score)
            self.train_scores_list.append(train_scores)
            self.test_scores_list.append(test_scores)
        for i in range(len(self.models_index)):
            print(f'==={self.models_index[i]}===')
            for j in range(5):
                print(f'test score: {self.test_scores_list[i][j]:.3f}   train score: {self.train_scores_list[i][j]:.3f}')

    def GetSupervised(self):
        """全ての手法のスコアをDataFrame形式で取得するメソッド
        Returns:
            pd.DataFrame: スコアを格納したDataFrame
        """
        self.df_scores = pd.DataFrame(
            self.test_scores_list,
            index=self.models_index)
        return self.df_scores.T

    def BestSupervised(self):
        """全ての手法のスコアをDataFrame形式で取得し、最も良いスコアを返すメソッド
        Returns:
            tuple: 最も良いスコアのインデックスとスコア
        """
        self.df_scores = pd.DataFrame(
            self.test_scores_list,
            index=self.models_index)
        self.df_scores['mean'] = self.df_scores.mean(axis=1)
        return self.df_scores['mean'].idxmax(), self.df_scores['mean'].max()

    def PlotFeatureImportanceAll(self):
        """全ての手法の特徴量の重要度をプロットするメソッド
        Returns:
            None
        """
        plot_models = [model for model in self.models if hasattr(
            model,
            'feature_importances_')
            ]
        for model in plot_models:
            self.visualizer.plot_feature_importances(
                feature_importances=model.feature_importances_,
                feature_names=self.feature_names,
                model_name=model.__class__.__name__)

    def VisualizeDecisionTree(self, figure_size=(16, 12)):
        """決定木の可視化を行うメソッド
        Args:
            figure_size (tuple, optional): 図のサイズ. Defaults to (16, 12).
        Returns:
            None
        """
        tree_model = next(model for model in self.models if isinstance(
            model,
            DecisionTreeClassifier
            ))
        plt.figure(figsize=figure_size)
        # 決定木の可視化
        plot_tree(
            tree_model,
            feature_names=list(self.x.columns),
            class_names=list(self.label_names),
            filled=True,
            impurity=False
            )
        plt.show()

    def PlotScaledData(self):
        """スケーリングしたデータをプロットするメソッド
        Returns:
            None
        """
        kfold = self.kfold
        X, y = self.x, self.y
        for fold, (train_index, test_index) in enumerate(kfold.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_score = []
            test_score = []
            X_train_scaled_list = [X_train.copy()]
            X_test_scaled_list = [X_test.copy()]
            # スケーリング方法の格納
            scaler = self.scaler
            # 用いるモデルの定義
            svm = LinearSVC(max_iter=5000, dual=False)
            svm.fit(X_train, y_train)
            train_score.append(svm.score(X_train, y_train))
            test_score.append(svm.score(X_test, y_test))
            for i in scaler:
                # 訓練データでスケーリング
                i.fit(X_train)
                X_train_scaled = i.transform(X_train)
                X_test_scaled = i.transform(X_test)
                # スケーリングしたデータをリストに追加
                X_train_scaled_list.append(pd.DataFrame(
                    X_train_scaled,
                    columns=X_train.columns,
                    index=X_train.index))
                X_test_scaled_list.append(pd.DataFrame(
                    X_test_scaled,
                    columns=X_test.columns,
                    index=X_test.index))
                # モデルのフィット
                svm.fit(X_train_scaled, y_train)
                train_score.append(svm.score(X_train_scaled, y_train))
                test_score.append(svm.score(X_test_scaled, y_test))

            for n in range(len(X_train.columns)):
                # スコアの出力
                print(f'Original     : test score: {test_score[0]:.3f}   train score: {train_score[0]:.3f}')
                for k in range(len(scaler)):
                    print(f'{scaler[k].__class__.__name__}   : test score: {test_score[k+1]:.3f}     train score: {train_score[k+1]:.3f}')
                # グラフの出力
                self.visualizer.plot_scaler_comparison(
                    X_train_scaled_list=X_train_scaled_list,
                    X_test_scaled_list=X_test_scaled_list,
                    X_label=X_train.columns[n],
                    y_label=X_train.columns[(n + 1) % len(X_train.columns)],
                    scaler=scaler)
                print('===================================================')

    def PlotFeatureHistgram(self):
        """特徴量のヒストグラムをプロットするメソッド
        Returns:
            None
        """
        X, y, df = self.x, self.y, self.df
        self.visualizer.plot_hist(
            X,
            y,
            df,
            label_names=self.label_names)

    def PlotPCA(self, n_components=2):
        """PCAをプロットするメソッド
        Args:
            n_components (int, optional): 主成分の数. Defaults to 2.
        Returns:
            tuple: スケーリングしたデータ、PCAの結果、PCAモデル
        """
        X, y = self.x, self.y
        label_names = self.label_names
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        df_X_pca = pd.DataFrame(X_pca)
        # グラフの描画
        self.visualizer.plot_scatter(
            X_pca,
            y,
            label_names)
        self.visualizer.plot_heatmap(pca.components_, X)
        return df_X_scaled, df_X_pca, pca

    def PlotNMF(self, n_components=2):
        """NMFをプロットするメソッド
        Args:
            n_components (int, optional): 主成分の数. Defaults to 2.
        Returns:
            tuple: スケーリングしたデータ、NMFの結果、NMFモデル
        """
        X, y = self.x, self.y
        label_names = self.label_names
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        nmf = NMF(
            n_components=n_components,
            random_state=0,
            max_iter=1000,
            init='nndsvd')
        nmf.fit(X)
        X_nmf = nmf.transform(X)
        df_X_nmf = pd.DataFrame(X_nmf)
        # グラフの描画
        self.visualizer.plot_scatter(
            X_nmf,
            y,
            label_names)
        self.visualizer.plot_heatmap(nmf.components_, X)
        return df_X_scaled, df_X_nmf, nmf

    def PlotTSNE(self, n_components=2):
        """t-SNEをプロットするメソッド
        Args:
            n_components (int, optional): 主成分の数. Defaults to 2.
        Returns:
            tuple: スケーリングしたデータ、t-SNEの結果、t-SNEモデル
        """
        X, y = self.x, self.y
        tsne = TSNE(n_components=n_components, random_state=0)
        X_tsne = tsne.fit_transform(X)
        X_tsne = pd.DataFrame(X_tsne, columns=[f'component_{i+1}' for i in range(n_components)])
        self.visualizer.plot_text(X_tsne, y)

    def PlotKMeans(self):
        """KMeans法をプロットするメソッド
        Returns:
            None
        """
        X = self.x
        y = np.array(self.y)
        # KMeansクラスタリングの適用
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        kmeans.fit(X)
        assignments = kmeans.labels_  # クラスタリング結果のラベル
        cluster_centers = kmeans.cluster_centers_  # クラスタの中心
        colors = ['blue', 'orange', 'green']
        markers = ['o', '*', 'D']
        print(f'KMeans法で予測したラベル: {kmeans.labels_}')
        self.visualizer.plot_clusters(
            X,
            y=assignments,
            model='KMeans',
            cluster_centers_=cluster_centers,
            centers=True,
            colors=colors,
            markers=markers)
        print(f'実際のラベル: {y}')
        self.visualizer.plot_clusters(
            X,
            y,
            model='KMeans',
            cluster_centers_=cluster_centers,
            colors=colors,
            markers=markers)

    def PlotDendrogram(self, truncate=False):
        """デンドログラムをプロットするメソッド
        Args:
            truncate (bool, optional): トランケートするかどうか. Defaults to False. 
        Returns:
            None
        """
        X = self.x
        linkage_array = ward(X)
        truncate = truncate
        self.visualizer.plot_dendrogram(
            linkage_array,
            truncate=truncate)

    def PlotDBSCAN(self, scaling=False, eps=1, min_samples=5):
        """DBSCANをプロットするメソッド
        Args:
            scaling (bool, optional): スケーリングするかどうか. Defaults to False.
            eps (int, optional): epsの値. Defaults to 1.
            min_samples (int, optional): min_samplesの値. Defaults to 5.
        Returns:
            None
        """
        X, y = self.x, self.y
        if scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = pd.DataFrame(X, columns=self.x.columns, index=self.x.index)
        else:
            X = X.copy()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        print(f'Cluster membership: {clusters}')
        self.visualizer.plot_clusters(
            X=X,
            y=y,
            model='DBSCAN',
            cluster_centers_=None,
            c=clusters)
