#!/usr/bin/env bash
set -euo pipefail

mkdir -p results38

run_case() {
  local ref="$1"
  local prompt="$2"
  local neg="$3"

  echo "===== Running ${ref} / config 1: ttc + msa ====="
  CUDA_VISIBLE_DEVICES=1 python examples/wan_14b_text_to_video.py \
    --mode effi_AMF \
    --seed 42 \
    --ttc_enabled \
    --msa_enabled \
    --output_dir results38 \
    --denoising_strength 0.88 \
    --msa_mask_mode amf \
    --height 480 --width 832 \
    --num_frames 81 \
    --input_video "data/${ref}.mp4" \
    --prompt "$prompt" \
    --negative_prompt "$neg"

  echo "===== Running ${ref} / config 2: ttc only ====="
  CUDA_VISIBLE_DEVICES=1 python examples/wan_14b_text_to_video.py \
    --mode effi_AMF \
    --seed 42 \
    --ttc_enabled \
    --output_dir results38 \
    --denoising_strength 0.88 \
    --msa_mask_mode amf \
    --height 480 --width 832 \
    --num_frames 81 \
    --input_video "data/${ref}.mp4" \
    --prompt "$prompt" \
    --negative_prompt "$neg"

  echo "===== Running ${ref} / config 3: msa only ====="
  CUDA_VISIBLE_DEVICES=1 python examples/wan_14b_text_to_video.py \
    --mode effi_AMF \
    --seed 42 \
    --msa_enabled \
    --output_dir results38 \
    --denoising_strength 0.88 \
    --msa_mask_mode amf \
    --height 480 --width 832 \
    --num_frames 81 \
    --input_video "data/${ref}.mp4" \
    --prompt "$prompt" \
    --negative_prompt "$neg"

  echo "===== Running ${ref} / config 4: baseline ====="
  CUDA_VISIBLE_DEVICES=1 python examples/wan_14b_text_to_video.py \
    --mode effi_AMF \
    --seed 42 \
    --output_dir results38 \
    --denoising_strength 0.88 \
    --msa_mask_mode amf \
    --height 480 --width 832 \
    --num_frames 81 \
    --input_video "data/${ref}.mp4" \
    --prompt "$prompt" \
    --negative_prompt "$neg"
}

run_case "ref5" "A giant prehistoric pterodactyl with highly detailed leathery brown and grey textured wings, flapping its wings vigorously and flying upwards into the sky, clear daytime sky, modern airport tarmac below, cinematic lighting, photorealistic scales, strong upward motion." "slow motion freeze, deformed wings, bird feathers, airplane, cartoon style, watermark, low quality"
run_case "ref2" "Back view of a heavy black steam locomotive train with intricate brass details and thick white smoke, slowly driving away on rusty tracks, desert canyon environment, golden sunset lighting, cinematic atmosphere, realistic mechanical motion." "floating train, deformed wheels, cartoon style, modern train, low quality"
run_case "ref1" "Back view of a futuristic astronaut in a white and gold spacesuit running toward a glowing lava lake, dark volcanic terrain, flying embers, intense red-orange lighting, cinematic sci-fi motion." "deformed body, extra limbs, cartoon style, low quality"
run_case "ref4" "A sleek silver high-speed bullet train moving smoothly along tracks, twilight sky with purple and orange hues, lakeside hills, glowing windows, cinematic motion." "flying train, deformed tracks, cartoon style, low quality"
run_case "ref6" "A humanoid robot walking up an escalator, metallic body with glowing blue energy lines, industrial mecha hangar environment, side tracking camera, steady motion." "blurry, static, extra limbs, deformed, low quality"
run_case "ref7" "A snow-white husky running across a snowy field, fur flowing with motion, snow particles flying, bright daylight, mountains in background, side tracking shot." "blurry, static, extra limbs, deformed, low quality"
run_case "ref8" "A heavy armored tank advancing across a desert, sand dust exploding behind tracks, strong forward motion, desert dunes, cinematic realism." "blurry, static, deformed tank, low quality"
run_case "ref9" "A motorcyclist speeding along a winding mountain road, leaning into turns, green hills and blue sky, dynamic side tracking shot." "blurry, static, deformed bike, low quality"
run_case "ref10" "A police officer with a German Shepherd walking up an escalator in a modern building, strong forward motion, busy environment, cinematic tracking from behind." "blurry, static, deformed body, low quality"
run_case "ref11" "FPV riding on a fire-breathing dragon flying fast, dark red scales, glowing magical forest, strong forward motion, fantasy cinematic scene." "blurry, static, deformed dragon, low quality"
run_case "ref13" "A futuristic robot lifting a heavy barbell, mechanical joints under tension, industrial gym environment, strong lifting motion, rear tracking view." "blurry, static, deformed robot, low quality"
run_case "ref14" "A lunar rover driving steadily across the moon surface, metallic structure, harsh sunlight, space background, strong forward motion." "blurry, static, deformed rover, low quality"
run_case "ref15" "Three motorcyclists riding side-by-side on blue motorcycles, synchronized motion, open road, bright daylight, side tracking shot." "blurry, static, deformed bikes, low quality"
run_case "source" "A brown horse galloping across grassland, powerful quadruped motion, flowing mane, golden sunset, cinematic side tracking." "floating horse, extra legs, cartoon style, low quality"

run_case "ref16" "First-person FPV aerial flight over the summit of massive snow mountains, thick white snow covering every ridge and peak, heavy snowfall drifting through cold misty air, snow particles floating in the wind, vast alpine landscape completely blanketed in bright white snow, cinematic winter atmosphere, smooth forward flying camera motion, realistic mountain terrain, high detail snow textures, cold blue lighting." "low quality, blurry snow, melting mountains, distorted terrain, floating landscape, camera shake, frozen motion, cartoon snow, unrealistic lighting, text overlay, watermark"
run_case "ref17" "First-person FPV flight crossing over a vast snowy mountain range, enormous peaks covered with thick white snow, snowflakes slowly falling through humid mountain air, soft fog drifting between valleys, endless winter landscape stretching to the horizon, smooth forward aerial motion, cinematic atmosphere, natural lighting, realistic alpine environment." "low resolution mountains, blurry terrain, distorted peaks, floating snow terrain, unrealistic fog, frozen motion, camera jitter, cartoon landscape, text overlay, watermark"
run_case "ref18" "First-person aerial orbit around the giant white statue of young Mao Zedong at Orange Isle in Changsha, the monumental sculpture standing proudly at the center of the island, surrounded by dense green trees and vegetation, the island encircled by the calm light-blue waters of the Xiang River, cinematic orbit camera motion, clear sky, realistic landscape lighting." "deformed statue, broken sculpture, duplicated statue, distorted face, floating statue, unrealistic island, messy trees, cartoon statue, text overlay, watermark, low quality"
run_case "ref19" "Side tracking camera following a powerful four-wheel ATV driving across a vast yellow-brown desert, large off-road tires kicking up clouds of sand and dust, rough desert terrain with endless dunes, strong sunlight casting sharp shadows, cinematic action scene, smooth lateral tracking motion, realistic dust particles and desert environment." "extra wheels, broken vehicle, floating ATV, distorted tires, unrealistic sand physics, frozen motion, duplicated vehicle, cartoon vehicle, text overlay, watermark, low quality"
run_case "ref20" "Cinematic orbiting aerial camera around the Statue of Liberty standing in the center of Liberty Island, the iconic green copper statue holding the torch high above New York Harbor, detailed sculpture texture and monumental scale, blue water surrounding the island, distant city skyline visible on the horizon, smooth circular camera motion, dramatic cinematic lighting." "multiple statues, distorted statue, broken monument, floating statue, unrealistic water, duplicated skyline, cartoon monument, camera glitch, text overlay, watermark, low quality"
run_case "ref21" "Cinematic orbit camera around a tall modern skyscraper standing at the center of a futuristic city, sleek glass architecture reflecting sunlight, surrounding urban skyline filled with modern buildings, clean geometric city design, smooth aerial orbit motion, bright daylight, realistic metropolitan environment." "distorted building, melting skyscraper, duplicated buildings, floating architecture, unrealistic reflections, broken city layout, cartoon city, camera glitch, text overlay, watermark, low quality"
run_case "ref22" "Macro close-up of a yellow-brown honeybee collecting nectar on a bright yellow flower, detailed fuzzy bee body and transparent wings visible, the bee slowly moving across the flower center covered with pollen, colorful flowers blooming in the background with soft blur, natural sunlight illuminating the scene, cinematic macro nature shot." "extra wings, deformed bee, multiple bees fused, floating insect, distorted flower, unrealistic colors, cartoon insect, frozen motion, text overlay, watermark, low quality"
run_case "ref23" "Wide cinematic countryside landscape showing three small European-style rural villas, traditional houses with sloped roofs, stone walls and wooden windows, surrounded by green fields and scattered trees, peaceful pastoral atmosphere, rolling countryside hills in the distance, soft warm sunlight, slow forward camera motion revealing the village scene." "extra houses, distorted buildings, melting architecture, floating houses, unrealistic countryside, cartoon village, broken roof shapes, duplicated structures, text overlay, watermark, low quality"

run_case "ref24" "Cinematic wildlife scene, a large wild goose soaring across the sky with powerful wing flapping, smooth gliding motion, sunset orange sky, volumetric lighting, wind affecting feathers, low-angle tracking shot, realistic flight dynamics." "frozen wings, hovering bird, extra wings, distorted anatomy, cartoon style, blur, watermark"
run_case "ref25" "Top-down aerial shot of a blue sedan driving steadily on the right lane of a highway, smooth forward motion, clean asphalt, no ocean, minimal environment, stable drone tracking." "floating car, ocean, wrong lane, distorted road, cartoon style, blur"
run_case "ref26" "Rear view of a British royal knight riding a white horse forward, holding a sword, medieval garden surroundings, steady motion, cinematic tracking from behind." "floating horse, distorted armor, modern objects, cartoon style"
run_case "ref27" "Side view of multiple AI robots running in a cyberpunk street at night, synchronized motion, neon lights, dynamic forward movement, tracking camera." "slow motion, frozen robots, floating, cartoon style"
run_case "ref28" "Six AI mech robots walking forward toward camera, heavy steps, cyberpunk armor, low-angle cinematic shot, strong presence." "floating robots, distorted bodies, cartoon style"
run_case "ref29" "A pack of gray wolves running together, coordinated motion, strong galloping rhythm, dust from ground, side tracking camera." "slow motion, floating wolves, extra legs, cartoon style"
run_case "ref30" "A British knight riding a white horse at full gallop, sword raised, strong motion, tiled ground reflections, side tracking shot." "slow motion, floating horse, distorted armor, cartoon style"
run_case "ref31" "Camera orbiting a giant futuristic spaceship in a modern city, smooth circular motion, reflective metallic surfaces, cinematic scale." "camera shake, distorted ship, low detail, cartoon style"
run_case "ref32" "Macro shot of purple morning glory flowers with a blue butterfly gently fluttering its wings while feeding nectar, soft lighting, shallow depth of field." "no motion, frozen butterfly, distorted wings, cartoon style"

run_case "ref33" "Cyberpunk cinematic chase, a single futuristic armored mech motorcycle carrying two humanoid AI robots, one driving in front and one sitting behind holding onto the rider, coordinated body leaning while turning at high speed, glowing neon blue and purple accents, reflective metallic bodies, motion blur trails and light streaks, futuristic skyscrapers with holographic billboards in background, night city atmosphere, low-angle side tracking shot following the motorcycle, strong sense of velocity and stable two-person riding interaction." "two motorcycles, separate vehicles, duplicated riders, floating characters, broken seating position, extra limbs, unrealistic physics, cartoon style, low detail, text overlay, watermark"
run_case "ref34" "Extreme close-up cinematic shot of human hands playing a silver-white piano, elegant slender fingers moving smoothly across keys, realistic finger articulation and rhythm, glossy metallic piano surface reflecting light, shallow depth of field, soft studio lighting, gentle camera micro-tracking following hand motion, natural motion continuity, warm and intimate atmosphere." "stiff fingers, broken joints, extra fingers, unrealistic motion, floating hands, blurry keys, cartoon style, low quality, text overlay, watermark"
run_case "ref35" "Cinematic warm interior scene, an elegant man in a black suit sitting on a chair and playing a silver-white piano, both hands moving gracefully and rapidly across the keys, expressive performance posture, neatly styled hair, refined facial features, soft golden ambient lighting, cozy environment, medium side angle shot slowly dollying in, smooth hand motion continuity, emotional and sophisticated atmosphere." "stiff body, unnatural hand motion, extra arms, distorted piano, broken anatomy, cartoon style, low detail, text overlay, watermark"
run_case "ref36" "Action cinematic scene, a firefighter in full protective orange firefighting gear running quickly forward on a city street while holding a high-pressure fire hose spraying water, helmet and reflective stripes flashing under urban light, realistic running dynamics, heavy gear movement, water particles scattering in the air, urgent forward momentum, handheld camera slightly shaking from a front-follow angle, strong motion continuity, emergency rescue atmosphere." "slow movement, floating character, broken legs, unrealistic water, duplicated body, cartoon style, low quality, text overlay, watermark"
run_case "ref37" "Military cinematic shot, a wheeled armored tank vehicle aggressively climbing a muddy dirt hillside, heavy suspension movement, spinning wheels throwing mud and thick yellow dust into the air, strong traction physics, rugged off-road terrain, large dust clouds trailing behind the vehicle, low rear three-quarter tracking camera angle, intense environmental interaction, realistic mechanical motion continuity, frame filled with yellow sand and dirt haze." "floating tank, unrealistic terrain, slow motion freeze, duplicated wheels, physics errors, cartoon style, low detail, text overlay, watermark"
run_case "ref38" "Dynamic superhero action scene, Spider-Man sprinting rapidly across a modern city park while pulling himself forward with stretched web lines, agile athletic body movement, visible web tension and acceleration, red and blue suit details, green lawns and park paths in the foreground, modern skyscrapers rising in the distant background, cinematic side tracking shot, strong motion blur and speed continuity, energetic and fluid movement." "stiff pose, broken limbs, floating character, unrealistic web physics, duplicated body, cartoonish style, low quality, text overlay, watermark"
run_case "ref39" "High-speed racing cinematic shot, a motorcycle rider wearing a full helmet leaning deeply into a sharp turn on a blue racing sport bike, aggressive drifting posture, fast cornering dynamics, tire friction sparks and smoke, realistic body lean and suspension compression, race track environment, low-angle side tracking camera closely following the rider, intense speed sensation, smooth motion continuity." "floating bike, unrealistic lean angle, broken rider pose, extra wheels, motion freeze, cartoon style, low detail, text overlay, watermark"

echo "===== All 38 cases completed ====="
