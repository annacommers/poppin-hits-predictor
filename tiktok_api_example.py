from TikTokApi import TikTokApi
api = TikTokApi()

results = 10

trending = api.discoverMusic()

for tiktok in trending:
    # Prints title of the song
    print(tiktok['cardItem']['title'])

print(len(trending))
