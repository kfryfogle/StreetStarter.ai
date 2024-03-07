
# StreetStarter.ai
AI Urban Development Application for CS 5100 - Foundations of AI

In this project, we aim to develop a program that is capable of generating city development plans based on user-specified dimensions. To generate the map, the user will input their desired dimensions and other parameters. Since we are not extracting data from a source to generate our initial map, the algorithm will initialize a map within these specifications. To ensure user-friendly interaction and visualization, we will construct a two-dimensional graphical user interface (GUI) using the PyGame library to showcase the generated plans. Our primary focus lies in optimizing street and building size allocation within cities of varying dimensions through the use of reinforcement learning.  With the provided map dimensions, we aim to maximize city_score which will be a combination of different scores such as area usage, connectivity, etc. Each city will include constraints such as connectivity, building types, building size, population density, etc. We will have multiple scenarios that will result in a punishment to the final score. For example, any of the cells surrounding a road that is not a building will include a punishment to the final reward score for the generated map. 

## Starter usage

- Install dependencies
- Run main.py
- press 'h' to select house, 't' for townhall, 'r' to rotate building and click on the grid to place it.
- Feel free to add more buildings types. 

## Next steps
- Develop the AI :)

## Version control
- Pull from main and create your own repo. Push changes in that and create a pull request to be merged to main.