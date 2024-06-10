# Define the classes and their respective groups
classes = {
    'backpack': 'backpack',
    'backpack_dog': 'backpack',
    'bear_plushie': 'stuffed animal',
    'berry_bowl': 'bowl',
    'can': 'can',
    'candle': 'candle',
    'cat': 'cat',
    'cat2': 'cat',
    'clock': 'clock',
    'colorful_sneaker': 'sneaker',
    'dog': 'dog',
    'dog2': 'dog',
    'dog3': 'dog',
    'dog5': 'dog',
    'dog6': 'dog',
    'dog7': 'dog',
    'dog8': 'dog',
    'duck_toy': 'toy',
    'fancy_boot': 'boot',
    'grey_sloth_plushie': 'stuffed animal',
    'monster_toy': 'toy',
    'pink_sunglasses': 'glasses',
    'poop_emoji': 'toy',
    'rc_car': 'toy',
    'red_cartoon': 'cartoon',
    'robot_toy': 'toy',
    'shiny_sneaker': 'sneaker',
    'teapot': 'teapot',
    'vase': 'vase',
    'wolf_plushie': 'stuffed animal'
}

# Define the prompt lists
object_prompts = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with the Eiffel Tower in the background',
    'a {0} {1} floating on top of water',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a white rug',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}'
]

live_subject_prompts = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} in a purple wizard outfit',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}'
]

# Function to write prompts to files
def write_prompts_to_file(classes, object_prompts, live_subject_prompts):
    for subject_name, class_name in classes.items():
        if class_name in ['backpack', 'bowl', 'can', 'candle', 'clock', 'sneaker', 'boot', 'glasses', 'cartoon', 'teapot', 'vase']:
            prompt_list = [prompt.format(subject_name, class_name) for prompt in object_prompts]
        else:
            prompt_list = [prompt.format(subject_name, class_name) for prompt in live_subject_prompts]

        file_name = f"prompts/{subject_name}_prompts.txt"
        with open(file_name, 'w') as file:
            file.write('\n'.join(prompt_list))
        print(f"Prompts written to {file_name}")

# Write the prompts to files
write_prompts_to_file(classes, object_prompts, live_subject_prompts)
