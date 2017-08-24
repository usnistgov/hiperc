/**********************************************************************************
 This file is part of Phase-field Accelerator Benchmarks, written by Trevor Keller
 and available from https://github.com/usnistgov/phasefield-accelerator-benchmarks.

 This software was developed at the National Institute of Standards and Technology
 by employees of the Federal Government in the course of their official duties.
 Pursuant to title 17 section 105 of the United States Code this software is not
 subject to copyright protection and is in the public domain. NIST assumes no
 responsibility whatsoever for the use of this software by other parties, and makes
 no guarantees, expressed or implied, about its quality, reliability, or any other
 characteristic. We would appreciate acknowledgement if the software is used.

 This software can be redistributed and/or modified freely provided that any
 derivative works bear some notice that they are derived from it, and any modified
 versions bear some notice that they have been modified.

 Questions/comments to Trevor Keller (trevor.keller@nist.gov)
 **********************************************************************************/

/**
 \file  output.c
 \brief Implementation of file output functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iso646.h>
#include <png.h>

#include "diffusion.h"

void print_progress(const int step, const int steps)
{
	/*
	Prints timestamps and a 20-point progress bar to stdout.
	Call inside the timestepping loop, near the top, e.g.

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);
		take_a_step();
		elapsed += dt;
	}
	*/

	char* timestring;
	static unsigned long tstart;
	struct tm* timeinfo;
	time_t rawtime;

	if (step==0) {
		tstart = time(NULL);
		time( &rawtime );
		timeinfo = localtime( &rawtime );
		timestring = asctime(timeinfo);
		timestring[strlen(timestring)-1] = '\0';
		printf("%s [", timestring);
		fflush(stdout);
	} else if (step==steps-1) {
		unsigned long deltat = time(NULL)-tstart;
		printf("•] %2luh:%2lum:%2lus\n",deltat/3600,(deltat%3600)/60,deltat%60);
		fflush(stdout);
	} else if ((20 * step) % steps == 0) {
		printf("• ");
		fflush(stdout);
	}
}

void write_csv(fp_t** conc, int nx, int ny, fp_t dx, fp_t dy, int step)
{
	int i, j;
	fp_t x, y;
	FILE* output;
	char name[256];
	char num[20];

	/* generate the filename */
	sprintf(num, "%07i", step);
	strcpy(name, "diffusion.");
	strcat(name, num);
	strcat(name, ".csv");

	/* open the file */
	output = fopen(name, "w");
	if (output == NULL) {
		printf("Error: unable to open %s for output. Check permissions.\n", name);
		exit(-1);
	}

	/* write csv data */
	fprintf(output, "x,y,c\n");
	for (j = 1; j < ny-1; j++) {
		y = dy * (j - 1);
		for (i = 1; i < nx-1; i++)	{
			x = dx * (i - 1);
			fprintf(output, "%f,%f,%f\n", x, y, conc[j][i]);
		}
	}

	fclose(output);
}

void write_png(fp_t** conc, int nx, int ny, int step)
{
	/* After "A simple libpng example program," http://zarb.org/~gc/html/libpng.html
	   and the libong manual, http://www.libpng.org/pub/png */

	fp_t min, max, *c;
	int i, j, w, h, n;
	FILE* output;
	char name[256];
	char num[20];
	unsigned char* buffer;

	png_infop info_ptr;
	png_bytepp row_pointers;
	png_structp png_ptr;
	png_byte color_type = PNG_COLOR_TYPE_GRAY;
	png_byte bit_depth = 8;

	w = nx - 2;
	h = ny - 2;

	/* generate the filename */
	sprintf(num, "%07i", step);
	strcpy(name, "diffusion.");
	strcat(name, num);
	strcat(name, ".png");

	/* open the file */
	output = fopen(name, "wb");
	if (output == NULL) {
		printf("Error: unable to open %s for output. Check permissions.\n", name);
		exit(-1);
	}

	/* allocate and populate image array */
	buffer = (unsigned char*)malloc(w * h * sizeof(unsigned char));
	row_pointers = (png_bytepp)malloc(h * sizeof(png_bytep));
	for (j = 0; j < h; j++)
		row_pointers[j] = &buffer[w * j];

	/* determine data range */
	min = 0.0;
	max = 1.0;
	for (j = ny-2; j > 0; j--) {
		for (i = 1; i < nx-1; i++) {
			c = &conc[j][i];
			if (*c < min)
				min = *c;
			if (*c > max)
				max = *c;
		}
	}

	/* rescale data into buffer */
	n = 0;
	for (j = ny-2; j > 0; j--) {
		for (i = 1; i < nx-1; i++) {
			buffer[n] = (unsigned char) 255 * (min + (conc[j][i] - min) / (max - min));
			n++;
		}
	}
	if (n != w * h) {
		printf("Error making image: expected %i values in buffer, got %i.\n", w*h, n);
		exit(-1);
	}

	/* let libpng do the heavy lifting */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		printf("Error making image: png_create_write_struct failed.\n");
		exit(-1);
	}
	info_ptr = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("Error making image: unable to init_io.\n");
		exit(-1);
	}
	png_init_io(png_ptr, output);

	/* write PNG header */
	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("Error making image: unable to write header.\n");
		exit(-1);
	}
	png_set_IHDR(png_ptr, info_ptr, w, h,
	                 bit_depth, color_type, PNG_INTERLACE_NONE,
	                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	/* write image */
	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("Error making image: unable to write data.\n");
		exit(-1);
	}
	png_write_image(png_ptr, row_pointers);

	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("Error making image: unable to finish writing.\n");
		exit(-1);
	}
	png_write_end(png_ptr, NULL);

	/* clean up */
	fclose(output);
	free(row_pointers);
	free(buffer);
}
