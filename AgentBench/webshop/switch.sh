#!/bin/bash

if [ -d "data_small" ] && [ -d "data" ]; then
    mv data data_all
    mv search_engine search_engine_all
    mv ./web_agent_site/utils.py ./web_agent_site/utils_all.py
    
    mv data_small data
    mv search_engine_small search_engine
    mv ./web_agent_site/utils_small.py ./web_agent_site/utils.py
    
    echo "all -> small "
elif [ -d "data_all" ] && [ -d "data" ]; then
    mv data data_small
    mv search_engine search_engine_small
    mv ./web_agent_site/utils.py ./web_agent_site/utils_small.py

    mv data_all data
    mv search_engine_all search_engine
    mv ./web_agent_site/utils_all.py ./web_agent_site/utils.py

    echo "small -> all "
else
    echo ""
fi
