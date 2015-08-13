library(plyr)
library(dplyr)
library(bit64)
library(data.table)
library(ggplot2)
library(caret)
setwd("/home/smizoe/Documents/kaggle/human_vs_robot/R")
bidders <- fread("./train.csv")
test.bidders <- fread("./test.csv")
bids  <- fread("./bids.csv")
threshold <- as.integer64("14000000000000")
set.seed(123)

inTrain <- as.integer(createDataPartition(bidders$outcome, p=0.8, list=F))
training  <- bidders[inTrain, ]
testing   <- bidders[-inTrain, ]


## explore
## inner join; see data in a user-wise manner. we can't see relation between bids
di <- inner_join(bids, training, by="bidder_id")

## # of bids matters (1)
cnts <- di[,.(num.bids=.N), by=.(bidder_id, outcome)]
qplot(factor(outcome), num.bids,data=cnts, geom="boxplot", log="y")

## check if robots do not sleep (2)
## 14000000000 is about with(bids, (max(time) - min(time)) / 10000)
#bid.time.hist <- mutate(di, hour=as.integer64(time)/as.integer64("14000000000"))[, .(count=.N), by=.(outcome, country, hour)]
#######bid.time.hist <- di[, .(count=.N), by=.(outcome, country, hour=as.integer(time%/%as.integer64("14000000000")))]
zero.time <- min(bids$time)
bid.time.hist = di[, .(count=.N), by=.(outcome,country, hour=as.integer((time - zero.time) %/% as.integer64(1000000000000)))]
bth.summary <- bid.time.hist[,.(count=sum(count)), by=.(outcome=as.factor(outcome),hour=hour)]
qplot(hour, count,data=bth.summary,facets=outcome~.,geom="bar",stat="identity")
qplot(hour, count,data=subset(bth.summary, hour<50),facets=outcome~.,geom="bar",stat="identity")
qplot(hour, count,data=subset(bth.summary, hour < 100& hour >= 50),facets=outcome~.,geom="bar",stat="identity")
qplot(hour, count,data=bth.summary,geom="line", colour=outcome)
qplot(hour, count, data=subset(bid.time.hist, country=="in"),facets=outcome~.,geom="bar",stat="identity")

## check if # of visitors differs by country
rb.by.country <- di[,.(count=.N),by=.(country, outcome)]
qplot(country,count,data=rb.by.country,geom="bar",stat="identity",fill=factor(outcome),position="dodge")+ theme(axis.text.x = element_text(angle = 90, hjust = 1))
rb.by.country[order(count,decreasing=T),]
#which( cumsum(rb.by.country[order(count,decreasing=T),]$count)/sum(rb.by.country[order(count,decreasing=T),]$count) > 0.95)[1]
#[1] 96

## check that there are countries with high bot rate
bot.rate <- ddply(rb.by.country, .(country),
      function(dt){
          vec  <- integer(0)
          for(num in 0:1){
              cand <- subset(dt, outcome== num)$count
              if(length(cand) == 0)
                  vec <- c(vec,0)
              else
                  vec <- c(vec,cand)
          }
          data.table(human=vec[1], bot=vec[2])
      })
qplot(human, bot, data=bot.rate, geom="text", label=country)
qplot(human, bot, data=subset(bot.rate, bot<20000), geom="text", label=country)

## check that there are referrers for which the rate of bots is higher than the others(3)
rb.by.url <- di[,.(human=sum(outcome==0), bot=sum(outcome==1),
                   human.uniq=length(unique(bidder_id[outcome==0])),
                   bot.uniq=length(unique(bidder_id[outcome==1]))
                   ), by=.(url)]
qplot(human, bot, data=rb.by.url)+geom_abline(slope=0.27)
qplot(human, bot, data=subset(rb.by.url,human <200000))+geom_abline(slope=0.27)
qplot(human.uniq, bot.uniq, data=rb.by.url)+geom_abline(slope=0.055)
qplot(human.uniq, bot.uniq, data=subset(rb.by.url, human.uniq< 1000))+geom_abline(slope=0.055)

#cum.data <- with(rb.by.url[order(bot+human),], data.table(id=length(url),urls=url,cum=cumsum(bot+human)))


## check that some categories attract more bots(4)
## (this is possibly because such categories are used in ads)

rb.by.cats <- di[,.(human=sum(outcome==0), bot=sum(outcome==1)), by=.(merchandise)]
qplot(human, bot, data=rb.by.cats)

## check that bots are more responsive to the other participants' bids(5)
## note that we can't use an inner-joined data, since we would like to use
## actual invervals between bids.
## (this might yield a conservative estimate, though)
## since time is obfuscated, we need to remove intervals that are too far apart;
## we filter out intervals that are more than 14000000000 * 1000
dl <- left_join(bids, training, by="bidder_id")
bid.interval <- ddply(dl, .(auction), function(dt){
                      dt <- dt[order(dt$time),] this.diff <- diff(dt$time) human     <- (dt$outcome == 0)[2:dim(dt)[1]]
                      bot       <- !human
                      human.val <- Filter(function(x) x < threshold, this.diff[which(human)])
                      bot.val   <- Filter(function(x) x < threshold, this.diff[which(bot)])
                      data.table(human.avg=mean(human.val), human.sd=sd(human.val),
                                 bot.avg=mean(bot.val), bot.sd=sd(bot.val))
      }
)
bid.interval$human.avg <- as.numeric(bid.interval$human.avg)
bid.interval$bot.avg <- as.numeric(bid.interval$bot.avg)
qplot(x=as.double(human.avg), y=as.double(bot.avg),data=bid.interval)
qplot(x=as.double(human.sd), y=as.double(bot.sd),data=bid.interval)
## seems to be not so useful
#ggplot(bid.interval, aes(xmin=human.avg-human.sd, xmax=human.avg + human.sd,
#                         ymin=bot.avg - bot.sd, ymax=bot.avg + bot.sd))+geom_rect()


## check that robots tend to have multiple IPs(6)
num.ip  <- di[, .(ips=length(unique(ip)),bids=.N,
                  auctions=length(unique(auction)),
                  categories=length(unique(merchandise)),
                  countries=length(unique(country)),
                  urls = length(unique(url))
                  ),
                      by = .(bidder_id, outcome)]
qplot(factor(outcome),ips, data=num.ip, geom="boxplot",log="y")+geom_jitter()
qplot(factor(outcome),bids, data=num.ip, geom="boxplot",log="y")+geom_jitter()
qplot(factor(outcome),auctions, data=num.ip, geom="boxplot",log="y")+geom_jitter()
qplot(factor(outcome),categories, data=num.ip, geom="boxplot")+geom_jitter()
qplot(factor(outcome),countries, data=num.ip, geom="boxplot")+geom_jitter()
qplot(factor(outcome),urls, data=num.ip, geom="boxplot")+geom_jitter()

## check if # of auctions participated differs between humans and bots(7)
num.auction <- di[, .(auctions=length(unique(auction))), by=.(bidder_id, outcome)]
qplot(factor(outcome), auctions, data=num.auction,geom="boxplot")+geom_jitter()

## check within-auction interval(7)
#calc <- function(fun, time){
#    tmp.val <- Filter(function(x) x < threshold,diff(sort(time)))
#    if(length(tmp.val) >0 )
#        fun(tmp.val)
#    else
#        NA
#}
#bid.interval.user.auction  <- di[, .(avg=calc(mean, time), sd=calc(sd, time)),by=.(bidder_id, outcome, auction)]
bid.interval.user.auction  <- di[, .(avg=mean(as.double(Filter(function(x) x < threshold,diff(sort(time))))),sd=sd(as.double(Filter(function(x) x < threshold,diff(sort(time)))))),by=.(bidder_id, auction, merchandise, outcome)]
qplot(factor(outcome),as.double(avg),data=bid.interval.user.auction, facets=.~merchandise, geom="jitter", log="y")+geom_boxplot()


## check within-user interval(8)
bid.interval.user  <- di[, .(avg=mean(as.double(Filter(function(x) x < threshold,diff(sort(time))))),sd=sd(as.double(Filter(function(x) x < threshold,diff(sort(time)))))),by=.(bidder_id, outcome)]
qplot(factor(outcome),as.double(avg),data=bid.interval.user,geom="jitter",log="y")+geom_boxplot()
qplot(factor(outcome),as.double(sd),data=bid.interval.user,geom="jitter",log="y")+geom_boxplot()


## check if devices matter


rb.by.device <- di[,.(human=sum(outcome==0), bot=sum(outcome==1),
                   human.uniq=length(unique(bidder_id[outcome==0])),
                   bot.uniq=length(unique(bidder_id[outcome==1]))
                   ), by=.(device)]
qplot(human, bot, data=rb.by.device)
#qplot(human, bot, data=subset(rb.by.device,human <200000))+geom_abline(slope=0.27)
#qplot(human.uniq, bot.uniq, data=rb.by.device)+geom_abline(slope=0.055)
#qplot(human.uniq, bot.uniq, data=subset(rb.by.device, human.uniq< 1000))+geom_abline(slope=0.055)
############
# feature creation
############

############
### feature mapping
############


char.to.int <- function(element, targets){
    if(element %in% targets)
        which(element == targets) - 1
    else
        length(targets)
}


############### bids
######### countries
all.countries <- sort(unique(bids$country))
cum.country.data <- mutate(bids[,.(count=.N),by=.(country)][order(count),], ratio=cumsum(count)/sum(count))
num.country <- dim(cum.country.data)[1]
target.countries <- cum.country.data$country[(num.country-98):num.country]

######### urls
## about 80% of urls have only one access => use urls with lots of accesses
all.urls <- sort(unique(bids$url))
cum.data <- mutate(bids[,.(count=.N),by=.(url)][order(count),], ratio=cumsum(count)/sum(count))

#> which(cum.data$count >= 1000)[1]
#[1] 1786176
#qplot(1:length(ratio), ratio,data=cum.data,geom="line")
num.url <- dim(cum.data)[1]
target.urls <- cum.data$url[-(1:1786175)]
#target.urls <- cum.data$url[(num.url - 98):num.url]



######### merchandises
all.merchandises  <- sort(unique(bids$merchandise))

######### devices
all.devices <- sort(unique(bids$device))

cum.dev.data <- mutate(bids[,.(count=.N),by=.(device)][order(count),], ratio=cumsum(count)/sum(count))
#> which(cum.dev.data$count >= 1000)[1]
#[1] 6757
#> dim(cum.dev.data)
#[1] 7351    3
num.dev <- dim(cum.dev.data)[1]
qplot(1:length(ratio), ratio,data=cum.dev.data,geom="line")
target.devices <- cum.dev.data$device[-c(1:6756)]
#target.devices <- cum.dev.data$device[(num.dev-98):num.dev]


############### bidders
######### addresses

all.addresses <- sort(unique(bidders$address))

######### payment_account
all.accounts  <- sort(unique(bidders$payment_account))

############
### per bidder features
############

per.bidder <- ddply(bids, .(bidder_id), function(df, thresh=threshold){
                    with(df,{
                         ## calculate time intervals between bids
                         intervals <- as.double(Filter(function(x) x<thresh,diff(sort(time))))
                         avg <- mean(intervals)
                         stdev <- sd(intervals)


                         data.table(
                                    ips=length(unique(ip)),
                                    bids=dim(df)[1],
                                    auctions=length(unique(auction)),
                                    avg.bid.interval=log(1+avg),
                                    sd.bid.interval=log(1+stdev)
                                    )
                          }
                    )
                  })
save(per.bidder, file="./per_bidder")
#load("./per_bidder")
bidders.df <- left_join(bidders, as.data.table(per.bidder), by="bidder_id")
test.bidders.df <- left_join(test.bidders, as.data.table(per.bidder), by="bidder_id")
bidders.df$bids <- as.numeric(bidders.df$bids)
bidders.df$ips <- as.numeric(bidders.df$ips)
bidders.df$auctions <- as.numeric(bidders.df$auctions)
bidders.df$outcome <- as.integer(bidders.df$outcome)
test.bidders.df$bids <- as.numeric(test.bidders.df$bids)
test.bidders.df$ips <- as.numeric(test.bidders.df$ips)
test.bidders.df$auctions <- as.numeric(test.bidders.df$auctions)

##### remove bidders with no bid data
bidders.df <- bidders.df[!is.na(bids),]


############
### imputation
############

knnImp <-  function(df, model=NULL){
    name <- names(df)
    target.indices <- which(!(sapply(df, class) %in% c("character", "factor","integer")))
    targets <- df[,name[target.indices], with=F]
    non.targets <- df[,name[-target.indices], with=F]
    if(is.null(model)){
        res <- preProcess(targets,method="knnImpute")
    }else{
        res  <- model
    }
    df.part <- predict(res, targets)
    df <- cbind(non.targets,df.part)
    list(df=df, result=res)
}

set.seed(123)
inTrain <- as.integer(createDataPartition(bidders.df$outcome, p=0.8, list=F))
training  <- bidders.df[inTrain, ]
testing   <- bidders.df[-inTrain, ]
inValidation <- createFolds(training$outcome)
cv.sets <- lapply(inValidation,
        function(indices){
            actual.train <- training[-indices,]
            validation   <- training[indices,]
            imputed.t <- knnImp(actual.train)
            imputed.v <- knnImp(validation, imputed.t$result)
            imputed.t$validation <- imputed.v$df
            imputed.t
        })
knn.by.train <- knnImp(training)
training.set <- knnImp(training, knn.by.train$result)
testing.set <- knnImp(testing, knn.by.train$result)$df

## fill feature values for 70 no-bid users
for(column in c("ips", "bids", "auctions")){
    replacement.targets <- which(is.na(test.bidders.df[, column, with=F]))
    col.indx  <- which(colnames(test.bidders.df) == column)
    set(test.bidders.df, i=replacement.targets, j=col.indx ,value=1)
}
knn.for.submission <- knnImp(bidders.df)
submission.training.set <- knnImp(bidders.df, knn.for.submission$result)$df
submission.set <- knnImp(test.bidders.df, knn.for.submission$result)$df
###########
### write data so that it is readable by spark
###########

### make mappings for categorical variables

for(name in c("countries", "merchandises","addresses","accounts")){
    factor.list <- get(paste("all",name,sep="."))
    write(paste(factor.list, 1:length(factor.list) - 1, sep="\t"),file=paste("./mappings/",name,".map",sep=""),sep="\n")
}

for(name in c("urls",  "devices")){
    factor.list <- get(paste("target",name,sep="."))
    write(paste(factor.list, 1:length(factor.list) - 1, sep="\t"),file=paste("./mappings/",name,".map",sep=""),sep="\n")
}

## training and validation sets
exploration.dir <- "data/exploration/"
indx <- 1
for(cv.set in cv.sets){
    write.table(cv.set$df, paste(exploration.dir, "bidders_train_", indx, ".csv", sep=""),quote=F,row.names=F, col.names=F, sep=",")
    write.table(cv.set$validation, paste(exploration.dir, "bidders_validation_", indx, ".csv", sep=""),quote=F,row.names=F, col.names=F, sep=",")
    indx <- indx+1
}

## testing sets
write.table(testing.set, paste(exploration.dir, "bidders_test.csv", sep=""), quote=F, row.names=F, col.names=F, sep=",")

## submission sets
submission.dir <- "data/submission/"
write.table(submission.training.set,paste(submission.dir, "bidders_submission_train.csv", sep=""), quote=F, row.names=F, col.names=F, sep=",")
write.table(submission.set,paste(submission.dir, "bidders_submission.csv", sep=""), quote=F, row.names=F, col.names=F, sep=",")

## create schemas for tables
## necessary schemas:
##   - bidders table
##   - bids table
##   - submission bidders table

convert.type <- Vectorize(function(type){
    if(type=="numeric")
        "double"
    else if(type == "character")
        "string"
    else if(type == "integer64")
        "decimal"
    else if(type == "integer")
        "int"
    else
        stop(paste("unknown type:", type))
})

mk.schema <- function(df){
    sig <- convert.type(sapply(df,class))
    col.names <- gsub("\\.", "_", names(df))
    paste(col.names, sig, sep="\t")
}
schema.dir <- "./schema/"
write(mk.schema(bidders.df),paste(schema.dir, "bidders.schema",sep=""),sep="\n")
write(mk.schema(bids),paste(schema.dir, "bids.schema",sep=""),sep="\n")
write(mk.schema(submission.set),paste(schema.dir, "submission.schema",sep=""),sep="\n")

######### Yet Another feature engineering

##### ip turns out to be not useful, since one ip corresponds to just one bot bidder.
ipdata.hoge <- di[,.(humans=sum(outcome==0), bots=sum(outcome==1), distinct.humans= length(unique(bidder <- id[outcome== 0])), distinct.bots=length(unique(bidder <- id[outcome== 1])) ),by=.(ip)]
ipdata <- mutate(ipdata.hoge, negentropy={prob<-bots/(humans+bots) ; ifelse(prob==0|prob==1, 0, prob * log2(prob) + (1-prob) * log2(1-prob))})
table(subset(ipdata,humans < bots & humans+bots> 100 & negentropy > -0.1)$distinct.bots)


################### define utility function
find.good.feature <- function(x){
    tmp <- di[,.(humans=sum(outcome==0), bots=sum(outcome==1), distinct.humans= length(unique(bidder_id[outcome== 0])), distinct.bots=length(unique(bidder_id[outcome== 1])) ),by=x]

    tmp <- mutate(tmp, negentropy={prob<-bots/(humans+bots) ; ifelse(prob==0|prob==1, 0, prob * log2(prob) + (1-prob) * log2(1-prob))})
###> 0.8 * log2(0.8) + 0.2 * log2(0.2)
###[1] -0.7219281
    print(table(subset(tmp,humans < bots & humans+bots> 100 & negentropy > -0.7)$distinct.bots))
    tmp
}

####> di[outcome==1, .N]
####[1] 383337
####> di[outcome==0, .N]
####[1] 2215312

######## device seems to work

device.data<- find.good.feature(quote(device))
## about half is covered if we use 286 devices
#length(subset(device.data,humans < bots & humans+bots> 100 & negentropy > -1)$bots)
#[1] 286
#sum(subset(device.data,humans < bots & humans+bots> 100 & negentropy > -1)$bots)
#[1] 189390
#> table(subset(device.data,humans < bots & humans+bots> 100 & negentropy > -1)$distinct.bots)
#
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 20 22 23 24 26 27 28 47
# 7 13 24 44 44 24 20 17 22 10 19 11  3  2  4  3  2  4  3  1  1  3  1  2  1  1
target.devices <- subset(device.data,humans < bots & humans+bots> 100 & negentropy > -1)$x

########## url seems not to work...
url.data<- find.good.feature(quote(url))
#> table(subset(url.data,humans < bots & humans+bots> 100 & negentropy > -1)$distinct.bots)
#
# 1  2
#54 12
