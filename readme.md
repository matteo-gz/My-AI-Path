## build site
```shell
hugo new site My-AI-Path
cd My-AI-Path
git init
touch .gitignore
touch Makefile
git submodule add https://github.com/alex-shpak/hugo-book themes/hugo-book
echo "theme = 'hugo-book'" >> hugo.toml
hugo server -D
```
