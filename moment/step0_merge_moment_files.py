import pandas as pd

# 合并多个动态文件
m1 = pd.read_json("../big_json/moments/moment_25.json", orient="records")
m2 = pd.read_json("../big_json/moments/moment_24.json", orient="records")
m3 = pd.read_json("../big_json/moments/moment_23.json", orient="records")
ms = pd.concat([m1, m2, m3])

# 合并多个点赞文件
l1 = pd.read_json("../big_json/moments/like_25_Sum.json", orient="records")
l2 = pd.read_json("../big_json/moments/like_25_Spr.json", orient="records")
l3 = pd.read_json("../big_json/moments/like_25_Fal.json", orient="records")
l4 = pd.read_json("../big_json/moments/like_24_Fal.json", orient="records")
l5 = pd.read_json("../big_json/moments/like_24_Spr.json", orient="records")
l6 = pd.read_json("../big_json/moments/like_24_Win.json", orient="records")
l7 = pd.read_json("../big_json/moments/like_24_Sum.json", orient="records")
l8 = pd.read_json("../big_json/moments/like_23_Fal.json", orient="records")
l9 = pd.read_json("../big_json/moments/like_23_Spr.json", orient="records")
l10 = pd.read_json("../big_json/moments/like_23_Win.json", orient="records")
l11 = pd.read_json("../big_json/moments/like_23_Sum.json", orient="records")

ls = pd.concat([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])

# 合并多个点赞文件
c1 = pd.read_json("../big_json/moments/comment_25_Sum_1.json", orient="records")
c2 = pd.read_json("../big_json/moments/comment_25_Spr_1.json", orient="records")
c3 = pd.read_json("../big_json/moments/comment_25_Fal_1.json", orient="records")
c4 = pd.read_json("../big_json/moments/comment_24_Fal_1.json", orient="records")
c5 = pd.read_json("../big_json/moments/comment_24_Spr_1.json", orient="records")
c6 = pd.read_json("../big_json/moments/comment_24_Win_1.json", orient="records")
c7 = pd.read_json("../big_json/moments/comment_24_Sum_1.json", orient="records")
c8 = pd.read_json("../big_json/moments/comment_23_Fal_1.json", orient="records")
c9 = pd.read_json("../big_json/moments/comment_23_Spr_1.json", orient="records")
c10 = pd.read_json("../big_json/moments/comment_23_Win_1.json", orient="records")
c11 = pd.read_json("../big_json/moments/comment_23_Sum_1.json", orient="records")

cs = pd.concat([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

print(len(ms))
print(len(ls))
print(len(cs))
print(ms.shape)
print(ls.shape)
print(cs.shape)

# ms.to_json("../big_json/moments/moments_merged.json", orient="records")
# ls.to_json("../big_json/moments/likes_merged.json", orient="records")
# cs.to_json("../big_json/moments/comments_merged.json", orient="records")
mstail = ms.tail(30)
mstailIds = mstail["_id"].tolist()
mstail.to_json("../aig_json/moments/moments_merged.json", orient="records")
ls[ls['momentId'].isin(mstailIds)].to_json("../aig_json/moments/likes_merged.json", orient="records")
cs[cs['momentId'].isin(mstailIds)].to_json("../aig_json/moments/comments_merged.json", orient="records")
