# barely works

from google_images_download import google_images_download

search_queries = [
    'yeji itzy',
    'lia itzy',
    'ryujin itzy',
    'chaeryeong itzy',
    'yuna itzy'
]
num_images = 5

for query in search_queries:
    response = google_images_download.googleimagesdownload()
    arguments = {'keywords': query, 'limit': num_images}
    paths = response.download(arguments)
    print(paths)
