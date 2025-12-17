# uva-machine-learning-25f-projects

For those who haven't submitted your project code yet, please follow the instructions below to upload your work to the course repository.

Step 1: Set up your local branch
- Go to the course repository and click Fork: https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects
- Go to your new forked repository and clone it to your local environment:
  - git clone https://github.com/<your-username>/uva-machine-learning-25f-projects.git
- Navigate into the cloned folder and add the original repository as an upstream remote:
  - git remote add upstream https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects.git

Step 2: Prepare your code:
- For each team, please create a folder named `team-XX` corresponding to your team ID (e.g., team-1, team-11, team-111). 
- Inside this folder, include the following:
  - src/: A subfolder containing all source code.
  - data/: A subfolder with the data required to reproduce results.
    - Note: If the data cannot be uploaded, include a markdown file describing how to collect it.
  - `requirements.txt`: A file listing required packages. (Format [reference](https://pip.pypa.io/en/stable/reference/requirements-file-format/))
  - `README.md`: A markdown file describing the folder content. You can view an example [here](https://github.com/QData/TextAttack). Your README should include:
    - Project Title
    - Team ID and Members
    - Overview: A brief introduction to the project.
    - Usage: How to run the code to get core results.
    - (Optional) Setup: Instructions for environment setup (if non-trivial).
    - (Optional) Video: A link to your demo video with a brief description.
- You are also welcome to include additional files or documentation in the folder or README.md if they help people better understand your project and code.

Step 3: Upload your code
- Commit your changes (no requirements on the commit message)
  - git add .
  - git commit -m "upload project code by Team-XX"
- Push the changes to your fork
  - git push origin main
- On GitHub, navigate to your fork and open a pull request via: Pull requests → New pull request