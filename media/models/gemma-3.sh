#!/bin/bash

#C0="\x1b[38;2;029;161;242m"
#C1="\x1b[38;2;054;148;220m"
#C2="\x1b[38;2;079;136;199m"
#C3="\x1b[38;2;104;123;177m"
#C4="\x1b[38;2;129;111;156m"
#C5="\x1b[38;2;155;098;134m"
#C6="\x1b[38;2;180;086;113m"
#C7="\x1b[38;2;205;073;091m"
#C8="\x1b[38;2;230;061;070m"
#C9="\x1b[38;2;255;048;048m"

C0="\x1b[38;2;000;255;124m"
C1="\x1b[38;2;000;255;124m"
C2="\x1b[38;2;000;255;124m"
C3="\x1b[38;2;000;255;124m"
C4="\x1b[38;2;000;255;124m"
C5="\x1b[38;2;000;255;124m"
C6="\x1b[38;2;000;255;124m"
C7="\x1b[38;2;000;255;124m"
C8="\x1b[38;2;000;255;124m"
C9="\x1b[38;2;000;255;124m"

CT="\x1b[38;2;029;161;242m"
CD="\x1b[38;2;255;048;048m"

CC="\x1b[0m"
CG="\x1b[38;2;000;255;124m"
CY="\x1b[38;2;255;238;110m"

echo -e "
 ${CT}ggml-org/llama.cpp ${CG}                        1/2

   ${CC}Model:   ${CG}Gemma 3  |  ${CC}Sizes: ${CG}4B, 12B, 27B
   ${CC}Created: ${CG}Google   |

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
