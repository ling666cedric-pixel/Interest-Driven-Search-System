import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class PersonalizedSearchSystem:
    def __init__(self):
        print(">>> 系统初始化中...正在加载 MovieLens 数据集...")
        # 1. 加载数据 (Data Layer )
        # 真实文档库
        self.movies = pd.read_csv('movies.csv')
        # 真实用户行为日志
        self.ratings = pd.read_csv('ratings.csv')

        # 数据预处理：将标题和分类合并，作为检索内容
        # 这样搜 "Action" 能搜到，搜 "Toy Story" 也能搜到
        self.movies['content'] = self.movies['title'] + " " + self.movies['genres'].str.replace('|', ' ')

        # 2. 构建索引 (Retrieval Module )
        # 使用 TF-IDF (Term Frequency-Inverse Document Frequency)
        # 这是最经典的信息检索算法，写进报告里非常标准
        print(">>> 正在构建 TF-IDF 倒排索引...")
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['content'])

        # 3. 预计算用户画像 (User Profiling [cite: 50])
        # 根据用户历史评分，统计用户最喜欢的 Genre (分类)
        print(">>> 正在挖掘用户兴趣画像...")
        self.user_profiles = self._build_user_profiles()
        print(">>> 系统就绪！\n")

    def _build_user_profiles(self):
        """
        根据 ratings.csv 计算每个用户的偏好
        逻辑：如果用户给 Action 电影打了 5 分，那他对 Action 的兴趣分就高
        """
        profiles = {}
        # 为了演示速度，我们只处理前 600 个用户 (MovieLens Small 也就 600 个用户)
        active_users = self.ratings['userId'].unique()

        # 将电影数据合并到评分数据里，方便查分类
        merged = self.ratings.merge(self.movies, on='movieId')

        for uid in active_users:
            # 找到该用户评过分的所有电影
            user_history = merged[merged['userId'] == uid]
            # 只看他喜欢的 (评分 >= 4.0)
            liked_movies = user_history[user_history['rating'] >= 4.0]

            # 统计他喜欢的电影的高频词/分类
            if not liked_movies.empty:
                # 简单画像：把所有喜欢的电影分类拼在一起，算出高频词
                all_genres = " ".join(liked_movies['genres'].str.replace('|', ' ').tolist())
                profiles[uid] = all_genres
            else:
                profiles[uid] = ""  # 冷启动用户
        return profiles

    def search(self, query, user_id=None, top_k=10):
        """
        核心功能：检索 + 个性化排序
        """
        # --- A. 基础检索 (Baseline Retrieval)  ---
        # 1. 把查询词转换成向量
        query_vec = self.tfidf.transform([query])
        # 2. 计算余弦相似度 (Cosine Similarity)
        cosine_sim = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        # 获取初步相关的文档索引 (比如前 50 个，作为候选集)
        # 我们不直接返回 top_k，而是取更多候选，再重排序
        candidate_indices = cosine_sim.argsort()[:-51:-1]

        results = []
        for idx in candidate_indices:
            score = cosine_sim[idx]
            if score > 0:
                results.append({
                    'id': self.movies.iloc[idx]['movieId'],
                    'title': self.movies.iloc[idx]['title'],
                    'genres': self.movies.iloc[idx]['genres'],
                    'base_score': score,
                    'final_score': score  # 暂时等于基础分
                })

        # --- B. 个性化重排序 (Personalized Ranking)  ---
        if user_id and user_id in self.user_profiles:
            user_interest_str = self.user_profiles[user_id]

            # 如果用户有画像，我们进行加权
            if user_interest_str:
                for item in results:
                    # 简单算法：如果电影的分类 在 用户的兴趣字符串里 出现过
                    # 比如用户喜欢 "Sci-Fi Action"，电影是 "Action"，那就加分
                    movie_genres = item['genres'].split('|')
                    match_count = sum(1 for g in movie_genres if g in user_interest_str)

                    # 加权公式：基础分 * (1 + 0.2 * 匹配的标签数)
                    # 这里的参数 0.2 可以在报告中讨论
                    boost = 1 + (0.2 * match_count)
                    item['final_score'] = item['base_score'] * boost

        # --- C. 最终排序 ---
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]


# --- 用户界面 (CLI) [cite: 52] ---
def main():
    system = PersonalizedSearchSystem()

    while True:
        print("=" * 60)
        print("可选真实用户 ID (来自 MovieLens): 1 (喜欢动画/冒险), 85 (喜欢惊悚/科幻)")
        # 这里的 ID 1 和 85 是 MovieLens 数据集里典型的用户，你可以自己在 csv 里找别的
        uid_input = input("请输入模拟的用户 ID (直接回车为匿名用户/Baseline, 'q'退出): ").strip()

        if uid_input == 'q':
            break

        user_id = int(uid_input) if uid_input.isdigit() else None

        query = input("请输入搜索词 (例如 'Star Wars', 'Love', 'Toy'): ").strip()
        if not query: continue

        results = system.search(query, user_id)

        print(f"\n>>> 搜索结果 (Top {len(results)}):")
        print(f"{'Title':<45} | {'Genre':<20} | {'Score':<6}")
        print("-" * 80)
        for res in results:
            title = res['title'][:43] + ".." if len(res['title']) > 43 else res['title']
            print(f"{title:<45} | {res['genres']:<20} | {res['final_score']:.4f}")
        print("\n")


if __name__ == "__main__":
    main()