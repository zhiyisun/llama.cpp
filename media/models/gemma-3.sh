#!/bin/bash

CT="\x1b[48;2;255;255;255m\x1b[38;2;029;161;242m"
CD="\x1b[49m\x1b[38;2;255;048;048m"

CC="\x1b[49m\x1b[0m"
CG="\x1b[49m\x1b[38;2;000;255;124m"
CY="\x1b[49m\x1b[38;2;255;238;110m"

echo -e "
 ${CT}ggml-org/llama.cpp ${CG}                        1/2 

   ${CC}Model:   ${CG}Gemma 3  ${CC}|  Sizes: ${CG}4B, 12B, 27B
   ${CC}Creator: ${CG}Google   ${CC}|

   ${CC}Sizes:        ${CG}4B, 12B, 27B
   ${CC}Capabilities: ${CG}Text, Tools, Vision

   ${CC}Attention:      ${CG}SWA 1:4
   ${CC}Vision Encoder: ${CG}SigLIP
   ${CC}Extra:          ${CG}QAT

   ${CY}  > llama-cli    -hf ggml-org/gemma-3
   ${CY}  > llama-server -hf ggml-org/gemma-3

   ${CC}License: ${CD}Gemma Terms of Use

"

#| textimg --background 13,26,39,255 -o gemma-3.png -f /usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf
