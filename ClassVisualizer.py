from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class Visualizer:
    def __init__(self):
        """初期化メソッド"""
        pass

    def plot_feature_importances(
        self,
        feature_importances,
        feature_names,
        model_name
    ):
        """特徴量の重要性をプロットする
        Args:
            feature_importances (np.ndarray): 特徴量の重要性
            feature_names (list[str]): 特徴量の名前
            model_name (str): モデルの名前
        Returns:
            None
        """
        indices = range(len(feature_importances))
        plt.barh(indices, feature_importances, align='center')
        plt.yticks(indices, feature_names)
        plt.xlabel(f'Feature Importance: {model_name}')
        plt.show()

    def tree_plot(
        self,
        tree_model,
        feature_names,
        class_names,
        figsize,
        filled=True,
        impurity=False
    ):
        """決定木をプロットする
        Args:
            tree_model (sklearn.tree): 決定木モデル
            feature_names (list[str]): 特徴量の名前
            class_names (list[str]): クラスの名前
            figsize (tuple): 図のサイズ
            filled (bool, optional): 塗りつぶすかどうか. Defaults to True.
            impurity (bool, optional): 不純度を表示するかどうか. Defaults to False.
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        plt.title('Decision Tree')
        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            impurity=impurity)
        plt.show()

    def plot_scaler_comparison(
        self,
        X_train_scaled_list,
        X_test_scaled_list,
        X_label,
        y_label,
        scaler,
        figsize=(15, 5)
    ):
        """スケーラーの比較をプロットする
        Args:
            X_train_scaled_list (list[pd.DataFrame]): スケーリングされたトレーニングデータ
            X_test_scaled_list (list[pd.DataFrame]): スケーリングされたテストデータ
            X_label (str): x軸のラベル
            y_label (str): y軸のラベル
            scaler (list[sklearn.preprocessing]): スケーラーのリスト
            figsize (tuple, optional): 図のサイズ. Defaults to (15, 5).
        Returns:
            None
        """
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        axes[0].scatter(
            X_train_scaled_list[0].iloc[:, 0],
            X_train_scaled_list[0].iloc[:, 1],
            label='Train set')
        axes[0].scatter(
            X_test_scaled_list[0].iloc[:, 0],
            X_test_scaled_list[0].iloc[:, 1],
            label='Test set',
            marker='^')
        axes[0].legend(loc='best')
        axes[0].set_title('Original Data')
        axes[0].set_xlabel(X_label)
        axes[0].set_ylabel(y_label)
        for i in scaler:
            axes[scaler.index(i) + 1].scatter(
                X_train_scaled_list[scaler.index(i) + 1].iloc[:, 0],
                X_train_scaled_list[scaler.index(i) + 1].iloc[:, 1],
                label='Train set')
            axes[scaler.index(i) + 1].scatter(
                X_test_scaled_list[scaler.index(i) + 1].iloc[:, 0],
                X_test_scaled_list[scaler.index(i) + 1].iloc[:, 1],
                label='Test set', marker='^')
            axes[scaler.index(i) + 1].set_title(f'{i.__class__.__name__}')
            axes[scaler.index(i) + 1].set_xlabel(X_label)
            axes[scaler.index(i) + 1].set_ylabel(y_label)
        plt.tight_layout()
        plt.show()

    def plot_hist(
        self,
        X,
        y,
        df,
        label_names,
        figsize=(10, 20),
        ylabel='Frequency'
    ):
        """ヒストグラムを描画する
        Args:
            X (pd.DataFrame): ヒストグラムを描画するデータ
            y (pd.Series): ヒストグラムを描画するデータのラベル
            df (pd.DataFrame): ヒストグラムを描画するデータ
            label_names (list[str]): ヒストグラムを描画するデータのラベルの名前
        Returns:
            None
        """
        fig, axes = plt.subplots(len(X.columns), 1, figsize=figsize)
        for i in range(len(X.columns)):
            _, bins = np.histogram(X.iloc[:, i], bins=50)
            for j in range(len(label_names)):
                axes[i].hist(
                    df[y == j].iloc[:, i],
                    bins=bins,
                    alpha=0.5,
                    label=label_names[j])
            axes[i].set_xlabel(X.columns[i])
            axes[i].set_ylabel(ylabel)
            axes[i].legend()
        plt.show()

    def plot_clusters(
        self,
        X,
        y,
        model,
        # label,
        cluster_centers_,
        centers=False,
        figsize=(8, 6),
        c=None,
        cmap='brg',
        colors=['blue', 'orange', 'green'],
        markers=['o', '^', 'v'],
        pointsize=100,
        lw=2,
        xlabel='Feature 2',
        ylabel='Feature 3',
        index_1=2,
        index_2=3
    ):
        """クラスタリング結果をプロットする
        Args:
            X (pd.DataFrame): プロットするデータ
            y (np.ndarray): クラスタリング結果
            model (str): モデル名
            cluster_centers_ (np.ndarray): クラスタ中心
            centers (bool, optional): クラスタ中心をプロットするかどうか. Defaults to False.
            figsize (tuple, optional): 図のサイズ. Defaults to (8, 6).
            c (np.ndarray, optional): 色. Defaults to None.
            cmap (str, optional): カラーマップ. Defaults to 'brg'.
            colors (list[str], optional): 色のリスト. Defaults to ['blue', 'orange', 'green'].
            markers (list[str], optional): マーカーのリスト. Defaults to ['o', '^', 'v'].
            pointsize (int, optional): ポイントサイズ. Defaults to 100.
            lw (int, optional): 線幅. Defaults to 2.
            xlabel (str, optional): x軸のラベル. Defaults to 'Feature 2'.
            ylabel (str, optional): y軸のラベル. Defaults to 'Feature 3'.
            index_1 (int, optional): x軸のインデックス. Defaults to 2.
            index_2 (int, optional): y軸のインデックス. Defaults to 3.
        Returns:
            None
        """
        if model == 'KMeans':
            plt.figure(figsize=figsize)
            for i, (color, marker) in enumerate(zip(colors, markers)):
                plt.scatter(
                    X.iloc[y == i, index_1],
                    X.iloc[y == i, index_2],
                    color=color,
                    marker=marker,
                    s=pointsize,
                    edgecolor='k')
            if centers:
                plt.scatter(
                    cluster_centers_[:, index_1],
                    cluster_centers_[:, index_2],
                    color='white',
                    edgecolors='k',
                    marker='^',
                    s=pointsize,
                    lw=lw)
        elif model == 'DBSCAN':
            plt.figure(figsize=figsize)
            plt.scatter(
                X.iloc[:, index_1],
                X.iloc[:, index_2],
                c=c,
                cmap=cmap,
                marker='o')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        else:
            raise ValueError(f"Unknown model: {model}")

    def plot_scatter(
        self,
        X,
        y,
        label_names,
        xlabel='First principal component',
        ylabel='Second principal component',
        figsize=(8, 8),
        colors=['blue', 'orange', 'green'],
        markers=['o', '^', 'v'],
        alpha=.8,
        lw=2,
        cmap='viridis'
    ):
        """散布図を描画する
        Args:
            X (np.ndarray): 散布図を描画するデータ
            y (np.ndarray): 散布図を描画するデータのラベル
            label_names (list[str]): 散布図を描画するデータのラベルの名前
            xlabel (str, optional): x軸のラベル. Defaults to 'First principal component'.
            ylabel (str, optional): y軸のラベル. Defaults to 'Second principal component'.
            figsize (tuple, optional): 図のサイズ. Defaults to (8, 8).
            colors (list[str], optional): 散布図の色. Defaults to ['blue', 'orange', 'green'].
            markers (list[str], optional): 散布図のマーカー. Defaults to ['o', '^', 'v'].
            alpha (float, optional): 散布図の透明度. Defaults to 0.8.
            lw (int, optional): 散布図の線幅. Defaults to 2.
            cmap (str, optional): カラーマップ. Defaults to 'viridis'.
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        label_list = np.arange(len(label_names))
        for color, marker, i, target_name in zip(
            colors,
            markers,
            label_list,
            label_names
        ):
            plt.scatter(
                X[y == i, 0],
                X[y == i, 1],
                color=color,
                alpha=alpha,
                lw=lw,
                marker=marker,
                label=target_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(label_names, loc='best')
        plt.gca().set_aspect('equal')
        plt.show()

    def plot_heatmap(
        self,
        components,
        X,
        xlabel="Feature",
        ylabel="Principal component",
        figsize=(8, 4),
        cmap='viridis'
    ):
        """ヒートマップを描画する
        Args:
            components (np.ndarray): ヒートマップを描画するデータ
            X (pd.DataFrame): ヒートマップを描画するデータ
            xlabel (str, optional): x軸のラベル. Defaults to "Feature".
            ylabel (str, optional): y軸のラベル. Defaults to "Principal component".
            figsize (tuple, optional): 図のサイズ. Defaults to (8, 4).
            cmap (str, optional): カラーマップ. Defaults to 'viridis'.
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        plt.matshow(components, cmap=cmap)
        plt.yticks([0, 1], ['First component', 'Second component'])
        plt.colorbar()
        plt.xticks(range(len(X.columns)), X.columns, rotation=60, ha='left')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_text(
        self,
        pd_data,
        y,
        color='black',
        fontsize=9,
        x_label='t-SNE feature 1',
        figsize=(8, 8)
    ):
        """散布図をラベルで描画する
        Args:
            pd_data (pd.DataFrame): 散布図を描画するデータ
            y (pd.Series): 散布図を描画するデータのラベル
            color (str, optional): テキストの色. Defaults to 'black'.
            fontsize (int, optional): フォントサイズ. Defaults to 9.
            x_label (str, optional): x軸のラベル. Defaults to 't-SNE feature 1'.
            figsize (tuple, optional): 図のサイズ. Defaults to (8, 8).
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        plt.xlim(pd_data.iloc[:, 0].min(), pd_data.iloc[:, 0].max() + 1)
        plt.ylim(pd_data.iloc[:, 1].min(), pd_data.iloc[:, 1].max() + 1)
        for i in range(len(pd_data)):
            plt.text(
                pd_data.iloc[i, 0],
                pd_data.iloc[i, 1],
                str(y.iloc[i]),
                color=color, fontsize=fontsize)
        plt.xlabel(x_label)
        plt.show()

    def plot_dendrogram(
        self,
        linkage_array,
        truncate=False,
        a=10,
        b=5.5,
        figsize=(10, 5),
        xlabel='sample index',
        ylabel='Cluster distance',
        truncate_mode='lastp',
        p=10
    ):
        """デンドログラムを描画する
        Args:
            linkage_array (np.ndarray): デンドログラムを描画するデータ
            truncate (bool, optional): デンドログラムをトランケートするかどうか. Defaults to False.
            a (float, optional): トランケートする高さ. Defaults to 10.
            b (float, optional): トランケートする高さ. Defaults to 5.5.
            figsize (tuple, optional): 図のサイズ. Defaults to (10, 5).
            xlabel (str, optional): x軸のラベル. Defaults to 'sample index'.
            ylabel (str, optional): y軸のラベル. Defaults to 'Cluster distance'.
            truncate_mode (str, optional): トランケートモード. Defaults to 'lastp'.
            p (int, optional): トランケートするサンプル数. Defaults to 10.
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        ax = plt.gca()
        if truncate:
            dendrogram(
                linkage_array,
                truncate_mode=truncate_mode,
                p=p)
        else:
            dendrogram(linkage_array)
            bounds = ax.get_xbound()
            ax.plot(bounds, [a, a], '--', c='k')
            ax.plot(bounds, [b, b], '--', c='k')
            ax.text(
                bounds[1],
                a,
                '3 clusters',
                va='center',
                ha='left',
                size=15)
            ax.text(
                bounds[1],
                b,
                '4 clusters',
                va='center',
                ha='left',
                size=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
