from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:8000/")

# state = [
#     (('/mnt/sfs-common/peiyuan/peiyuan/LongVa/local_demo/assets/otter_books.jpg',), None),
#     ("What does this image show?", None)
# ]
state = [ ("What's the video about?", None)]


# result = client.predict(
# 		history=[],
# 		message={'text': "What's the video about?", 'files': []},
# 		video_input="/mnt/sfs-common/peiyuan/peiyuan/LongVa/local_demo/cache/0333f1f38db26d76752fee6cd2d938f06617dd2f/dc_demo.mp4",
# 		api_name="/add_message"
# )

# result = client.predict(
# 		video_input=None,
# 		state=state,
# 		sample_frames=16,
# 		temperature=0.7,
# 		max_new_tokens=1024,
# 		top_p=1,
# 		api_name="/bot_response_1"
# )
# print(result)

image_path = "local_demo/assets/assistant_logo.png"
# image to base64
import base64
with open(image_path, "rb") as image_file:
	base64_image = base64.b64encode(image_file.read()).decode()
 
input_json_url = [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    },
    {
		"role": "assistant",
		"content": [
			{"type": "text", "text": "The image shows a serene, open field with lush green grass and a few scattered trees or shrubs in the distance. The sky above is mostly clear with some wispy clouds, suggesting it's a bright and likely pleasant day. There's a light-colored wooden boardwalk or path that meanders through the field, inviting a peaceful walk along its length. The overall scene conveys a sense of tranquility and natural beauty."},
		]
	},
    {
		"role": "user",
		"content": [
			{"type": "text", "text": "Where is this place?"},
		]
	}
  ]

input_json_base64 = [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}",
          },
        },
      ],
    }
]
result = client.predict(
		input_json=input_json_base64,
		api_name="/base64_api"
)
print(result)
