#!/bin/bash

function get_color_code() {
    case "$1" in
        red) echo "\e[31m" ;;
        green) echo "\e[32m" ;;
        yellow) echo "\e[33m" ;;
        blue) echo "\e[34m" ;;
        magenta) echo "\e[35m" ;;
        cyan) echo "\e[36m" ;;
        bold) echo "\e[1m" ;;
        *) echo "" ;;  # no color or unknown
    esac
}

function bold_status() {
    local flair="=========="
    local message="$1"
    local color="$2"
    local color_code
    color_code=$(get_color_code "$color")

    echo -e "${color_code}${flair} ${message} ${flair}\e[0m"
}

function status() {
    local flair=":::::"
    local message="$1"
    local color="$2"
    local color_code
    color_code=$(get_color_code "$color")

    echo -e "${color_code}${flair} ${message} ${flair}\e[0m"
}
