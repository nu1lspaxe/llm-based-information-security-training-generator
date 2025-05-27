# A LLM-based Information Security Training Generator

## Setup Steps

1. Start a neo4j database with docker

    ```bash
    docker run --name neo4j -d --publish=7474:7474 --publish=7687:7687 --volume=$($PWD.Path)/neo4j/data:/data neo4j
    ```

2. Replace neo4j user password 

    ```bash
    docker exec -it neo4j cypher-shell -u neo4j -p neo4j -d system "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'neo4jpasswd';"
    ```

3. Activate virutal python runtime

    ```bash
    ./.venv/Scripts/activate
    ```

4. Run the application

    ```bash
    python main.py
    ```