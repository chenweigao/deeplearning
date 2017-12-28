#include<stdio.h> 
#include<time.h>
#include<stdlib.h>
int a[10000],n=10000;
int compare=0,change=0;
int quicksort(int left,int right){
	int i,j,t,temp;
	if(left>right)
	return 0;
	temp=a[left];
	i=left;
	j=right;
	while(i!=j){
		while(a[j]>=temp&&i<j){
		   compare++;
		   j--;
		}
		while(a[i]<=temp&&i<j)
		{
		    i++;
		    compare++;
		}
		if(i<j){
			t=a[i];a[i]=a[j];a[j]=t;change++;
		}
	}
	a[left]=a[i];
    a[i]=temp;
    quicksort(left,i-1);
    quicksort(i+1,right);
}

int main()
{
  int i;
  for(i=0;i<10000;i++)
      a[i]=1+(int)rand();
      
  FILE *fp1=fopen("infile.txt","w");
  if(fp1=NULL) return-1;
  for(i=0;1<10000;i++)
  	  fprintf(fp1,"%d",a[i]);
  fclose(fp1);
  
  quicksort(0,n-1);
  printf("比较次数；%d ",compare);
  printf("交换次数；%d",change);
 // printf("比较时间：%d",time);
  return 0;
  	
}

