This note is a guidance to set up and work with Hadoop MapReduce scripts on a
Pseudo Distributed Hadoop system. In order to successfully run these codes, I
need to set up the Hadoop system first.

### Step 1: Grab some sample data from GitHub

After the Hadoop Pseudo Distributed system is up and running, ssh into the
instance as you would normally do. We will need to pull some data and some java scripts from GitHub to work with the HDFS system and MapReduce. Run from your instance's $Home directory:

`git clone https://github.com/bindatype/HadoopProblem.git`

My $Home directory now has the following files:

```shell
ubuntu@ip-172-31-15-183:~$ ls -al
total 40
drwxr-xr-x  7 ubuntu ubuntu 4096 Dec 14 16:36 .
drwxr-xr-x  3 root   root   4096 Dec 14 15:38 ..
-rw-r--r--  1 ubuntu ubuntu  220 Aug 31  2015 .bash_logout
-rw-r--r--  1 ubuntu ubuntu 3912 Dec 14 16:22 .bashrc
drwx------  2 ubuntu ubuntu 4096 Dec 14 15:40 .cache
drwxr-xr-x 13 ubuntu ubuntu 4096 Dec 14 16:29 hadoop-2.7.4
drwxrwxr-x  5 ubuntu ubuntu 4096 Dec 14 16:36 HadoopProblem
drwxrwxr-x  2 ubuntu ubuntu 4096 Dec 14 15:44 .nano
-rw-r--r--  1 ubuntu ubuntu  655 May 16  2017 .profile
drwx------  2 ubuntu ubuntu 4096 Dec 14 15:59 .ssh
-rw-r--r--  1 ubuntu ubuntu    0 Dec 14 15:45 .sudo_as_admin_successful
```

### Step 2: Move the data directory into hdfs

The HDFS system is a completely different file system from ubuntu's file system. Therefore, for the Hadoop / MapReduce scripts to work with desired files, one needs to move the file into the HDFS system. I ran (from within the ~/hadoop-2.7.4 directory) the following command to put the folder into the HDFS system:

`bin/hdfs dfs -put ~/HadoopProblem/data/googlebooks-eng-all-1gram-20090715-tiny.tsv input`

This script tells the Hadoop to fire up the hdfs system and copy the file over to input folder within the HDFS system. I then entered the following code to confirm the existing files in the HDFS system:

`bin/hdfs dfs -ls input`

I get:

```shell
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hdfs dfs -ls input
Found 1 items
-rw-r--r--   1 ubuntu supergroup        103 2017-12-14 19:37 input/googlebooks-eng-all-1gram-20090715-tiny.tsv
```

So the file is successfully copied over to the HDFS system. I then move on to the next stage.

### Step 3: Compile java codes into classes and create a jar file for MapReduce

**Note:** The underlying java scripts are already tested and ready for this step. In real life scenario, you certainly need to write your own MapReduce functions before compiling them for use

From the ~/hadoop-2.7.4 directory, run the following command to compile all .java files into a class:

`bin/hadoop com.sun.tools.javac.Main ~/HadoopProblem/mapreduce/AggregateJob.java ~/HadoopProblem/mapreduce/ProjectionMapper.java`

Class files will be created **in the directories where the .java files lie**. Relocate to that directory and you shall see:

```shell
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hadoop com.sun.tools.javac.Main ~/HadoopProblem/mapreduce/AggregateJob.java ~/HadoopProblem/mapreduce/ProjectionMapper.java
Note: /home/ubuntu/HadoopProblem/mapreduce/AggregateJob.java uses or overrides a deprecated API.
Note: Recompile with -Xlint:deprecation for details.
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ cd ~/HadoopProblem/mapreduce/
ubuntu@ip-172-31-15-183:~/HadoopProblem/mapreduce$ ls
AggregateJob.class  AggregateJob.java  ProjectionMapper.class  ProjectionMapper.java
```

After confirming the .class files are generated, go ahead and create a .jar file by running `$jar cf jar_name.jar` that put the classes into one place. This will be our MapReduce blueprint in next steps. Results shown below:

```shell
ubuntu@ip-172-31-15-183:~/HadoopProblem/mapreduce$ jar cf pm.jar AggregateJob.class ProjectionMapper.class
ubuntu@ip-172-31-15-183:~/HadoopProblem/mapreduce$ ls
AggregateJob.class  AggregateJob.java  pm.jar  ProjectionMapper.class  ProjectionMapper.java
```

### Step 4: Run the jar file by calling function AggregateJob on the data directory and output it to a directory named output_final

Now that the MapReduce blueprint is created, it's time to run the .jar on the file we put in the HDFS system. The structure of a MapReduce command is as below:

`hadoop_directory$ bin/hadoop jar jar_file.jar MainFunctionName hdfs_input_directory hdfs_output_directory`

On our specific case, we will have:

```shell
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hadoop jar ~/HadoopProblem/mapreduce/pm.jar AggregateJob /user/ubuntu/input user/ubuntu/output_final
17/12/14 19:47:10 INFO client.RMProxy: Connecting to ResourceManager at /0.0.0.0:8032
...
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hdfs dfs -ls user/ubuntu/output_final
Found 2 items
-rw-r--r--   1 ubuntu supergroup          0 2017-12-14 19:47 user/ubuntu/output_final/_SUCCESS
-rw-r--r--   1 ubuntu supergroup         24 2017-12-14 19:47 user/ubuntu/output_final/part-r-00000
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hdfs dfs -cat user/ubuntu/output_final/*
dobbs	42
doctor	1214191
```

### Step 5: Retrieve the output_final directory from HDFS onto the local file system

We might want to move the output files from the HDFS system back to ubuntu's system for our reporting purposes. From the Hadoop installation directory, run:

`$bin/hdfs dfs -get hdfs_directory local_directory`

We have:

```shell
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ bin/hdfs dfs -get user/ubuntu/output_final output_final
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ ls
bin  include  lib      LICENSE.txt  NOTICE.txt  output_final  sbin   src
etc  input    libexec  logs         output      README.txt    share
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ cd output_final/
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4/output_final$ ls
part-r-00000  _SUCCESS
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4/output_final$ cd ..
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ cat output_final/*
dobbs	42
doctor	1214191
ubuntu@ip-172-31-15-183:~/hadoop-2.7.4$ cd ~/HadoopProblem/
ubuntu@ip-172-31-15-183:~/HadoopProblem$ mkdir results
ubuntu@ip-172-31-15-183:~/HadoopProblem$ ls
data  mapreduce  README.md  results
ubuntu@ip-172-31-15-183:~/HadoopProblem$ cd results/
ubuntu@ip-172-31-15-183:~/HadoopProblem/results$ mv ~/hadoop-2.7.4/output_final/ .
ubuntu@ip-172-31-15-183:~/HadoopProblem/results$ ls
output_final
ubuntu@ip-172-31-15-183:~/HadoopProblem/results$ cat output_final/*
dobbs	42
doctor	1214191
```
