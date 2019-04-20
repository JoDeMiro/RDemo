
# install.packages("plot3D")
# install.packages("plot3Drgl")
# install.packages("rgl")

download.file('http://www.pintye.com/300XValtozo1YCSV_pontosvesszovel_tagolt.csv', 'adat.csv')

df = read.csv("adat.csv", header = TRUE, sep = ';', dec = '.')

head(df)

names(df)

length(names(df))

scatter.smooth(x=df$x1, y=df$y, main="x1 ~ y")

scatter.smooth(x=df$x2, y=df$y, main="x2 ~ y")

hist(df$x1)
hist((df$y))

par(mfrow=c(1, 2))  # divide graph area in 2 columns
boxplot(df$x1, main="x1", sub=paste("Outlier rows: ", boxplot.stats(df$x1)$out))
boxplot(df$x2, main="x2", sub=paste("Outlier rows: ", boxplot.stats(df$x2)$out))

linearMod <- lm(y ~ x1, data = df)
print(linearMod)
summary(linearMod)
cor(linearMod$fitted.values, df$y)


linearMod <- lm(y ~ x2, data = df)
print(linearMod)
summary(linearMod)
cor(linearMod$fitted.values, df$y)

linearMod <- lm(y ~ x1 + x2, data = df)
print(linearMod)
summary(linearMod)
cor(linearMod$fitted.values, df$y)



library("plot3D")

x <- sep.l <- df$x1
y <- pet.l <- df$y
z <- sep.w <- df$x2





hist3D_fancy<- function(x, y, break.func = c("Sturges", "scott", "FD"), breaks = NULL,
                        colvar = NULL, col="white", clab=NULL, phi = 5, theta = 25, ...){
  
  # Compute the number of classes for a histogram
  break.func <- break.func [1]
  if(is.null(breaks)){
    x.breaks <- switch(break.func,
                       Sturges = nclass.Sturges(x),
                       scott = nclass.scott(x),
                       FD = nclass.FD(x))
    y.breaks <- switch(break.func,
                       Sturges = nclass.Sturges(y),
                       scott = nclass.scott(y),
                       FD = nclass.FD(y))
  } else x.breaks <- y.breaks <- breaks
  
  # Cut x and y variables in bins for counting
  x.bin <- seq(min(x), max(x), length.out = x.breaks)
  y.bin <- seq(min(y), max(y), length.out = y.breaks)
  xy <- table(cut(x, x.bin), cut(y, y.bin))
  z <- xy
  
  xmid <- 0.5*(x.bin[-1] + x.bin[-length(x.bin)])
  ymid <- 0.5*(y.bin[-1] + y.bin[-length(y.bin)])
  
  oldmar <- par("mar")
  par (mar = par("mar") + c(0, 0, 0, 2))
  hist3D(x = xmid, y = ymid, z = xy, ...,
         zlim = c(-max(z)/2, max(z)), zlab = "counts", bty= "g", 
         phi = phi, theta = theta,
         shade = 0.2, col = col, border = "black",
         d = 1, ticktype = "detailed")
  
  scatter3D(x, y,
            z = rep(-max(z)/2, length.out = length(x)),
            colvar = colvar, col = gg.col(100),
            add = TRUE, pch = 18, clab = clab,
            colkey = list(length = 0.5, width = 0.5,
                          dist = 0.05, cex.axis = 0.8, cex.clab = 0.8)
  )
  par(mar = oldmar)
}

hist3D_fancy(df$y, df$x1, 
             colvar=as.numeric(df$x2))

hist3D_fancy(df$y, df$x1, colvar=df$x2,
             breaks =30)

# Create his3D using plot3D
hist3D_fancy(iris$Sepal.Length, iris$Petal.Width, colvar=as.numeric(iris$Species))
# Make the rgl version
library("plot3Drgl")
plotrgl()











