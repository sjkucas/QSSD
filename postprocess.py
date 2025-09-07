import json
import copy

label1 = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
              'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
              'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
              'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
              'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

# label1 = ['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle',
#                     'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling',
#                     'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves',
#                     'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles',
#                     'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira',
#                     'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood',
#                     'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws',
#                     'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree',
#                     'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing',
#                     'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics',
#                     'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof',
#                     'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo',
#                     'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
#                     'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop',
#                     'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying',
#                     'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding',
#                     'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks',
#                     'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture',
#                     'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton',
#                     'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums',
#                     'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey',
#                     'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball',
#                     'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin',
#                     'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta',
#                     'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting',
#                     'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing',
#                     'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping',
#                     'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs',
#                     'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining',
#                     'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning',
#                     'Spread mulch','Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming',
#                     'Swinging at the playground', 'Table soccer','Tai chi', 'Tango', 'Tennis serve with ball bouncing',
#                     'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars',
#                     'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine',
#                     'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes',
#                     'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents',
#                     'Zumba']
with open(r'data\thumos14\thumos14.json', 'r', encoding='utf-8') as f1, open(r'output.json', 'r', encoding='utf-8') as f2:
    data_dict = json.load(f1)
    data_dict1 = json.load(f2)

for filename, sub_dict in data_dict['database'].items():
    if 'val' in filename:
        if 'annotations' in sub_dict and isinstance(sub_dict['annotations'], list):
            sub_dict['annotations'] = []

for item in data_dict1['shuju']:
    data = {'label':label1[int(item[4])], 'segment':[round(item[1],2), round(item[2],2)], 'label_id': int(item[4]), 'score': round(item[3],4)}
    if data['score'] > 0.3:
        data_dict['database'][item[0]]['annotations'].append(data)

for filename1, sub1 in data_dict['database'].items():
    if 'val' in filename1:
        labels = {d["label"] for d in sub1['annotations']}
        label_ids = {d["label_id"] for d in sub1['annotations']}

        pairs = {(d["label"], d["label_id"]) for d in sub1['annotations']}
        if len(labels)>1:
            pairs = list(pairs)
            for d in sub1['annotations']:
                d['label']=pairs[0][0]
                d['label_id']=pairs[0][0]
            data = copy.deepcopy(sub1['annotations'])
            for d in data:
                d['label'] = pairs[1][0]
                d['label_id'] = pairs[1][1]
            sub1['annotations'].extend(data)
            data_dict['database'][filename1]['annotations'] = sub1['annotations']


with open(r'data\thumos14\thumos14_soft_pseudo.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)